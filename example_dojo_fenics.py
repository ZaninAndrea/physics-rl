from dojo import Dojo
from coordinated import (
    coordinated,
    stop_coordination,
    CoordinatedContext,
    CoordinatedPyEnvironment,
)
from tf_agents.environments import suite_gym, tf_py_environment, py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

from fenics import TimeProblem
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh


class HeatEnv(py_environment.PyEnvironment):
    def __init__(self, comm):
        self._episode_ended = False
        self.steps_count = 0

        # Create mesh and define function space
        nx, ny = 50, 50
        dt = 0.01
        domain = mesh.create_rectangle(
            comm,
            [np.array([-2, -2]), np.array([2, 2])],
            [nx, ny],
            mesh.CellType.triangle,
        )
        V = fem.FunctionSpace(domain, ("CG", 1))

        # Create boundary condition
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        bc = fem.dirichletbc(
            PETSc.ScalarType(1),
            fem.locate_dofs_topological(V, fdim, boundary_facets),
            V,
        )

        # Define variational problem
        u_n = fem.Function(V)
        u_n.name = "u_n"

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

        f = fem.Constant(domain, PETSc.ScalarType(0))
        L = (u_n + dt * f) * v * ufl.dx

        problem = TimeProblem(domain)
        problem.set_A(a, [bc])
        problem.set_L(L)
        problem.set_dt(dt)
        problem.set_initial_condition(lambda x: np.array(0.0 * x[0]))
        problem.set_u_n(u_n)
        self.problem = problem

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
        self.domain = domain

    def action_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name="action"
        )

    def observation_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float64, name="observation"
        )

    def _get_problem_observation(self):
        temperature_first_quadrant = self.domain.comm.allreduce(
            fem.assemble_scalar(
                fem.form(self.problem.u_n * self.first_quadrant * ufl.dx)
            ),
            op=MPI.SUM,
        )
        temperature_second_quadrant = self.domain.comm.allreduce(
            fem.assemble_scalar(
                fem.form(self.problem.u_n * self.second_quadrant * ufl.dx)
            ),
            op=MPI.SUM,
        )
        temperature_third_quadrant = self.domain.comm.allreduce(
            fem.assemble_scalar(
                fem.form(self.problem.u_n * self.third_quadrant * ufl.dx)
            ),
            op=MPI.SUM,
        )
        temperature_fourth_quadrant = self.domain.comm.allreduce(
            fem.assemble_scalar(
                fem.form(self.problem.u_n * self.fourth_quadrant * ufl.dx)
            ),
            op=MPI.SUM,
        )

        return np.array(
            [
                temperature_first_quadrant,
                temperature_second_quadrant,
                temperature_third_quadrant,
                temperature_fourth_quadrant,
            ]
        )

    def _reset(self):
        self.problem.reset()
        self.steps_count = 0
        self._episode_ended = False

        return ts.restart(self._get_problem_observation())

    def _step(self, action):
        if self._episode_ended or self.steps_count >= 10:
            return self.reset()

        self.steps_count += 1

        # Make sure episodes don't go on forever.
        if action == 0:
            self._episode_ended = True
        elif action == 1:
            self.problem.solve(0.1)
        else:
            raise ValueError("`action` should be 0 or 1.")

        if self._episode_ended or self.steps_count >= 10:
            reward = np.sum(self._get_problem_observation())
            return ts.termination(self._get_problem_observation(), reward)
        else:
            return ts.transition(
                self._get_problem_observation(), reward=-0.1, discount=1.0
            )


comm = MPI.COMM_WORLD

with CoordinatedPyEnvironment(comm, HeatEnv(comm)) as env:
    if comm.Get_rank() == 0:
        env = tf_py_environment.TFPyEnvironment(env)

        print("Running on {} processes".format(comm.size), flush=True)
        q_net = QNetwork(env.observation_spec(), env.action_spec())

        dojo = Dojo(q_net, env)
        dojo.train(100)
