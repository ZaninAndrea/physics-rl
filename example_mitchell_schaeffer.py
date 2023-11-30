from dojo import Dojo
from coordinated import (
    random,
    Coordinator,
    parallel,
)
from tf_agents.environments import tf_py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

from fem import MonodomainMitchellSchaeffer
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, mesh, cpp
import math
import typing


def pentaray_catheter_positions(
    distances: typing.List[float],
) -> typing.List[typing.List[float]]:
    positions = []

    alpha = 2 * math.pi / 5
    for d in distances:
        for i in range(5):
            x = d * math.cos(math.pi / 2 - alpha * i)
            y = d * math.sin(math.pi / 2 - alpha * i)

            positions.append([x, y])

    return positions


DEBUG = False
MAX_STEPS = 15
SUCCESS_REWARD = 500
SHOCK_REWARD = -10
SHOCK_INTENSITY = 50
ACTIVATION_VANISHING_THRESHOLD = 100
CATHETER_POSITIONS = pentaray_catheter_positions([37.5, 32.5, 17.5, 12.5])
CATHETER_DISTANCE_DECAY_RATE = 33


# half_plane returns a function with half the plane set to 0
# and the other half set to a linearly increasing value up to 1
def half_plane(center: float, fade_width: float = 20):
    def f(x):
        v = np.zeros_like(x[0])
        v[x[0] < center] = np.minimum(
            -x[0][x[0] < center] / fade_width, np.ones_like(x[0][x[0] < center])
        )

        return v

    return f


# zero returns a function that is always 0
def zero(x):
    return np.zeros_like(x[0])


# ICD_Shock returns a function that is 0 everywhere except
# in a round region centered in (cx, cy) with radius r
def ICD_Shock(cx=0, cy=25, r=25, soft_edges=True):
    def f(x):
        v = np.zeros_like(x[0])

        if soft_edges:
            v = np.exp(r**2 - ((x[0] - cx) ** 2 + (x[1] - cy) ** 2))
            v[v < 0] = 0
            v[v > SHOCK_INTENSITY] = SHOCK_INTENSITY
        else:
            v[(x[0] - cx) ** 2 + (x[1] - cy) ** 2 < r**2] = SHOCK_INTENSITY

        return v

    return f


# exponential_distance returns a function that decreases exponentially
# with distance from the point (cx, cy) with decay rate mu
def exponential_distance(V, cx, cy, mu):
    dist = fem.Function(V)
    dist.interpolate(
        lambda x: np.exp(-mu * np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2))
    )

    return dist


class System(MonodomainMitchellSchaeffer):
    def __init__(self, coordinator: Coordinator):
        self.comm = coordinator.MPI_Comm()

        # Setup equation parameters
        nx, ny = 60, 60
        domain = mesh.create_rectangle(
            self.comm,
            [np.array([-50, -50]), np.array([50, 50])],
            [nx, ny],
            mesh.CellType.quadrilateral,
        )

        V = fem.FunctionSpace(domain, ("CG", 1))

        super().__init__(
            coordinator,
            domain,
            zero,
            half_plane(0),
            dt=0.03,
        )

        # Setup problem observations
        self.cathether_distances = [
            exponential_distance(V, *pos, CATHETER_DISTANCE_DECAY_RATE)
            for pos in CATHETER_POSITIONS
        ]

        self.steps_count = 0

    def action_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name="action"
        )

    def observation_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(len(CATHETER_POSITIONS),), dtype=np.float64, name="observation"
        )

    def get_observation(self):
        observations = [
            self.compute_ufl_form(fem.form(self.u_n() * dist * ufl.dx))
            for dist in self.cathether_distances
        ]

        return np.array(observations)

    def get_total_activation(self):
        return self.compute_ufl_form(fem.form(self.u_n() * ufl.dx))

    # Reset the environment, this method is called automatically be tf-agents
    @parallel
    def _reset(self):
        self.steps_count = 0

        # Test random initialization until we get a valid one
        while True:
            # Randomly initialize w
            half_plane_center = (random(self.comm) * 20) - 10
            self.set_initial_condition(zero, half_plane(half_plane_center))
            self.reset_w_n()

            # Reset u
            self._problem.reset()

            # Simulate a shock with random width and bottom centered on the w activation
            width = (random(self.comm) * 3) + 3
            bottom = random(self.comm) * 30

            def shock_I_app(x):
                v = np.array(0.0 * x[0])
                v[
                    np.logical_and(
                        x[1] > bottom, np.abs(x[0] - half_plane_center) < width
                    )
                ] = SHOCK_INTENSITY

                return v

            self.set_I_app(shock_I_app)
            self.advance_time(15)

            # Check if the simulation is valid
            if not math.isnan(self.get_total_activation()):
                break
            elif DEBUG:
                print("Nan detected, retrying...", flush=True)

            # Stop the shock and simulate the spiral evolution
            self.set_I_app(zero)
            self.advance_time(85)

        if DEBUG:
            self.save_u_image("u.png")

        return ts.restart(self.get_observation())

    @parallel
    def _step(self, action):
        if self.steps_count >= MAX_STEPS:
            return self.reset()

        self.steps_count += 1

        # Apply shock or not depending on action
        if action == 0:
            self.set_I_app(zero)
            self.advance_time(50)
        elif action == 1:
            self.set_I_app(ICD_Shock())
            self.advance_time(25)
            self.set_I_app(zero)
            self.advance_time(25)
        else:
            raise ValueError("`action` should be 0 or 1.")

        if DEBUG:
            self.save_u_image("u.png")

        # Stop early if activation has vanished
        activation = self.get_total_activation()
        if activation < ACTIVATION_VANISHING_THRESHOLD:
            if DEBUG:
                print("Activation vanished early", flush=True)
            return ts.termination(
                self.get_observation(), SUCCESS_REWARD - self.steps_count
            )

        # Stop agent interaction after MAX_STEPS steps
        if self.steps_count >= MAX_STEPS:
            # Compute observation before simulating more steps to
            # keep the time step consistent, the additional simulations
            # are used only to compute the reward
            observation = self.get_observation()

            # Simulate 20 more steps to check if the activation vanishes
            # without applying any agent interaction
            for i in range(20):
                self.steps_count += 1
                self.advance_time(50)

                if DEBUG:
                    self.save_u_image("u.png")

                activation = self.get_total_activation()

                if activation < ACTIVATION_VANISHING_THRESHOLD:
                    if DEBUG:
                        print("Activation vanished", flush=True)

                    return ts.termination(
                        observation, SUCCESS_REWARD - self.steps_count
                    )

            if DEBUG:
                print("Activation remained", flush=True)

            return ts.termination(observation, 0)
        else:
            return ts.transition(
                self.get_observation(), reward=SHOCK_REWARD * action, discount=1.0
            )


# Create the environment and the agent, then
# train the agent on the environment
coordinator = Coordinator()
env = System(coordinator)
with coordinator:
    if coordinator.is_leader():
        env = tf_py_environment.TFPyEnvironment(env)
        q_net = QNetwork(
            env.observation_spec(), env.action_spec(), fc_layer_params=(100, 100)
        )

        dojo = Dojo(q_net, env)

        dojo.train(100)
