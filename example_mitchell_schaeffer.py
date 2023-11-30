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

SHOCK_INTENSITY = 50
MAX_STEPS = 15


# half_plane returns a function with half the plane set to 0
# and the other half set to a linearly increasing value up to 1
def half_plane(center: float):
    def f(x):
        v = np.array(0.0 * x[0])
        v[x[0] < center] = np.minimum(
            -x[0][x[0] < center] / 20, np.ones_like(x[0][x[0] < center])
        )

        return v

    return f


# zero returns a function that is always 0
def zero(x):
    return np.zeros_like(x[0])


# ICD_Shock returns a function that is 0 everywhere except
# in a round region centered in (25,25)
def ICD_Shock(x):
    v = np.array(0.0 * x[0])

    cx = 0
    cy = 25
    r = 25
    v[(x[0] - cx) ** 2 + (x[1] - cy) ** 2 < r**2] = SHOCK_INTENSITY

    return v


# ICD_Shock returns a function that is 0 everywhere except
# in a round region centered in (25,25)
def circle(x):
    cx = 0
    cy = 0
    r = 10

    return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 < r**2


class System(MonodomainMitchellSchaeffer):
    def __init__(self, coordinator: Coordinator):
        self.comm = coordinator.MPI_Comm()

        # Setup equation parameters
        nx, ny = 80, 80
        domain = mesh.create_rectangle(
            self.comm,
            [np.array([-50, -50]), np.array([50, 50])],
            [nx, ny],
            mesh.CellType.quadrilateral,
        )

        V = fem.FunctionSpace(domain, ("CG", 1))
        self._half_plane_center = 0

        super().__init__(
            coordinator,
            domain,
            zero,
            half_plane(self._half_plane_center),
            dt=0.03,
        )

        # Setup problem observations
        self.first_quadrant = fem.Function(V)
        self.first_quadrant.interpolate(
            lambda x: np.logical_and(x[0] > 0, x[1] > 0).astype(float)
        )
        self.second_quadrant = fem.Function(V)
        self.second_quadrant.interpolate(
            lambda x: np.logical_and(x[0] < 0, x[1] > 0).astype(float)
        )
        self.third_quadrant = fem.Function(V)
        self.third_quadrant.interpolate(
            lambda x: np.logical_and(x[0] < 0, x[1] < 0).astype(float)
        )
        self.fourth_quadrant = fem.Function(V)
        self.fourth_quadrant.interpolate(
            lambda x: np.logical_and(x[0] > 0, x[1] < 0).astype(float)
        )

        self.steps_count = 0

    def action_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name="action"
        )

    def observation_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float64, name="observation"
        )

    def get_observation(self):
        temperature_first_quadrant = self.compute_ufl_form(
            fem.form(self.u_n() * self.first_quadrant * ufl.dx)
        )
        temperature_second_quadrant = self.compute_ufl_form(
            fem.form(self.u_n() * self.second_quadrant * ufl.dx)
        )
        temperature_third_quadrant = self.compute_ufl_form(
            fem.form(self.u_n() * self.third_quadrant * ufl.dx)
        )
        temperature_fourth_quadrant = self.compute_ufl_form(
            fem.form(self.u_n() * self.fourth_quadrant * ufl.dx)
        )

        return np.array(
            [
                temperature_first_quadrant,
                temperature_second_quadrant,
                temperature_third_quadrant,
                temperature_fourth_quadrant,
            ]
        )

    # Reset the environment, this method is called automatically be tf-agents
    @parallel
    def _reset(self):
        self.steps_count = 0

        # Randomly initialize w
        self._half_plane_center = (random(self.comm) * 20) - 10
        self.set_initial_condition(zero, half_plane(self._half_plane_center))
        # self.reset_w_n()

        # Reset u
        self._problem.reset()

        # Simulate a shock with random with centered on the w activation
        width = (random(self.comm) * 7) + 3

        def shock_I_app(x):
            v = np.array(0.0 * x[0])
            v[
                np.logical_and(x[1] > 0, np.abs(x[0] - self._half_plane_center) < width)
            ] = SHOCK_INTENSITY

            return v

        self.set_I_app(shock_I_app)
        self.advance_time(15)

        # Stop the shock and simulate the spiral evolution
        self.set_I_app(zero)
        self.advance_time(85)

        self.save_u_image("u.png")

        return ts.restart(self.get_observation())

    @parallel
    def _step(self, action):
        if self.steps_count >= MAX_STEPS:
            return self.reset()

        self.steps_count += 1

        # Apply shock or not depending on action
        if self.steps_count == 8:
            action = 1
        else:
            action = 0

        if action == 0:
            self.set_I_app(zero)
            self.advance_time(30)
        elif action == 1:
            self.set_I_app(ICD_Shock)
            self.advance_time(10)
            self.set_I_app(zero)
            self.advance_time(20)
        else:
            raise ValueError("`action` should be 0 or 1.")

        self.save_u_image("u.png")

        # Compute reward
        if self.steps_count >= MAX_STEPS:
            observation = self.get_observation()
            reward = np.sum(observation)
            return ts.termination(observation, reward)
        else:
            return ts.transition(self.get_observation(), reward=-0.1, discount=1.0)


# Create the environment and the agent, then
# train the agent on the environment
coordinator = Coordinator()
env = System(coordinator)
with coordinator:
    if coordinator.is_leader():
        env = tf_py_environment.TFPyEnvironment(env)
        q_net = QNetwork(env.observation_spec(), env.action_spec())

        dojo = Dojo(q_net, env)

        dojo.train(100)
