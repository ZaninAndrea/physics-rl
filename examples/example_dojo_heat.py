from dojo import Dojo
from coordinated import (
    randint,
    Coordinator,
    parallel,
)
from tf_agents.environments import tf_py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

from fem import HeatEnvironment
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, mesh


class System(HeatEnvironment):
    def __init__(self, coordinator: Coordinator):
        self.comm = coordinator.MPI_Comm()

        # Setup heat equation parameters
        nx, ny = 50, 50
        domain = mesh.create_rectangle(
            self.comm,
            [np.array([-2, -2]), np.array([2, 2])],
            [nx, ny],
            mesh.CellType.triangle,
        )

        dirichlet_bc = PETSc.ScalarType(1)
        initial_condition = lambda x: np.array(0.0 * x[0])
        V = fem.FunctionSpace(domain, ("CG", 1))
        source_term = fem.Constant(domain, PETSc.ScalarType(0))

        super().__init__(
            coordinator,
            domain,
            dirichlet_bc,
            initial_condition,
            source_term,
            V=V,
            dt=0.01,
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
        self._episode_ended = False

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

    @parallel
    def handle_reset(self):
        self.steps_count = 0
        self._episode_ended = False

        level = randint(self.comm, -5, 5)
        self.set_initial_condition(lambda x: np.array(0.0 * x[0] + level))

    @parallel
    def _step(self, action):
        if self._episode_ended or self.steps_count >= 10:
            return self.reset()

        self.steps_count += 1

        # Make sure episodes don't go on forever.
        if action == 0:
            self._episode_ended = True
        elif action == 1:
            self.advance_time(0.1)
        else:
            raise ValueError("`action` should be 0 or 1.")

        if self._episode_ended or self.steps_count >= 10:
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
