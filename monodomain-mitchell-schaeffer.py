from petsc4py import PETSc
from dolfinx import fem, io, plot
import pyvista
import matplotlib.pyplot as plt
from tf_agents.environments import suite_gym, tf_py_environment, py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from coordinated import (
    random,
    Coordinator,
    Coordinated,
    parallel,
)
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from abc import ABC, abstractmethod
from typing import Any
from fenics import TimeProblem
from ufl import conditional
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from time import sleep
from random import randint


# MonodomainMitchellSchaeffer is an abstract base class that implements a
# monodomain Mitchell-Schaeffer environment for reinforcement learning.
# The PDE defining the environment and the solution can be modified
# during the simulation to account for interactions with the agent.
class MonodomainMitchellSchaeffer(
    # py_environment.PyEnvironment,
    Coordinated,
    ABC,
):
    def __init__(
        self,
        coordinator: Coordinator,
        domain: Any,
        dirichlet_bc: Any,
        initial_condition: Any,
        tau_in: float = 0.3,
        tau_out: float = 6.0,
        tau_open: float = 75.0,
        tau_close: float = 80.0,
        u_gate: float = 0.13,
        D: float = 0.013,
        dt: float = 0.01,
        V: Any = None,
    ) -> None:
        self.register_coordinator(coordinator)

        # Parse model parameters
        self.dt = dt
        self.domain = domain

        if V is None:
            V = fem.FunctionSpace(domain, ("CG", 1))
        self._V = V

        self._tau_in = tau_in
        self._tau_out = tau_out
        self._tau_open = tau_open
        self._tau_close = tau_close
        self._u_gate = u_gate
        self._D = D
        self.intensity = 50

        # Define variational problem
        self._u, self._v, self._u_n, self._w_n, self._I_app = (
            ufl.TrialFunction(V),
            ufl.TestFunction(V),
            fem.Function(V, name="u_n"),
            fem.Function(V, name="w_n"),
            fem.Function(V, name="I_app"),
        )
        self._u_n.name = "u_n"
        self._w_n.name = "w_n"
        self._problem = TimeProblem(domain)

        self.compute_right_hand_side()
        self.set_dirichlet_bc(dirichlet_bc)

        def half(x):
            v = np.array(0.0 * x[0])
            v[x[0] < 0] = 1

            return v

        self.set_initial_condition(initial_condition, half)
        self._problem.set_dt(dt)
        self._problem.set_u_n(self._u_n)

    # Get the current problem
    def problem(self) -> TimeProblem:
        return self._problem

    # Get the current transmembrane potential
    def u_n(self) -> Any:
        return self._u_n

    # Get the current gate variable
    def w_n(self) -> Any:
        return self._w_n

    # Reset the gate variable
    def reset_w_n(self) -> None:
        def half(x):
            v = np.array(0.0 * x[0])
            v[x[0] < 0] = 0.5

            return v

        self._w_n.interpolate(half)
        self.compute_right_hand_side()

    # Change the initial condition of the problem
    def set_initial_condition(self, initial_u: Any, initial_w) -> None:
        self._problem.set_initial_condition(initial_u)
        self._w_n.interpolate(initial_w)

    def compute_right_hand_side(self):
        def I_app(x):
            v = np.array(0.0 * x[0])
            v[np.logical_and(x[1] > 0, np.abs(x[0]) < 5)] = self.intensity

            return v

        self._I_app.interpolate(I_app)

        J_in = self._w_n * self._u_n * self._u_n * (1 - self._u_n) / self._tau_in
        J_out = -self._u_n / self._tau_out
        L = (self._u_n + self.dt * (J_in + J_out + self._I_app)) * self._v * ufl.dx

        self._problem.set_L(L)

    # Change the Dirichlet boundary condition of the problem
    def set_dirichlet_bc(self, dirichlet_bc):
        # Create boundary condition
        fdim = self.domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        bc = fem.dirichletbc(
            dirichlet_bc,
            fem.locate_dofs_topological(self._V, fdim, boundary_facets),
            self._V,
        )

        # Define variational problem
        a = (
            self._u * self._v * ufl.dx
            + self.dt * self._D * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
        )

        self._problem.set_A(a, [])
        # self._problem.set_A(a, [bc])

    # Compute the integral of a UFL form over the domain in parallel
    @parallel
    def compute_ufl_form(self, form):
        return self.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(form)),
            op=MPI.SUM,
        )

    # Advance the time by T in the problem
    @parallel
    def advance_time(self, T: float) -> None:
        t = 0
        while t < T:
            # Update the gate variable with forward euler
            current_w = self.w_n().copy()
            current_w.name = "w_n"
            current_u = self.u_n()

            update = conditional(
                current_u < self._u_gate,
                (1 - current_w) / self._tau_open,
                -current_w / self._tau_close,
            )
            new_w = fem.Expression(
                current_w + self.dt * update, self._V.element.interpolation_points()
            )
            self._w_n.interpolate(new_w)

            # Update transmembrane potential solving the variational problem
            self._problem.solve(self.dt)

            t += self.dt

    # handle_reset can be overridden to implement custom reset behavior,
    # it will be called after the problem is reset, but before
    # the observation is computed.
    def handle_reset(self):
        pass

    # Reset the environment, this method is called automatically be tf-agents
    @parallel
    def _reset(self):
        self.reset_w_n()
        self._problem.reset()
        self.handle_reset()

        return ts.restart(self.get_observation())

        # get_observation should be overridden to implement custom observations
        # @abstractmethod
        # def get_observation(self) -> Any:
        #     pass


nx, ny = 200, 200
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-50, -50]), np.array([50, 50])],
    [nx, ny],
    mesh.CellType.quadrilateral,
)

coordinator = Coordinator()
env = MonodomainMitchellSchaeffer(
    coordinator,
    domain,
    PETSc.ScalarType(0),
    lambda x: np.zeros_like(x[0]),
    dt=0.03,
)
with coordinator:
    if coordinator.is_leader():
        u_topology, u_cell_types, u_geometry = plot._(env._V)

        u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
        u_grid.point_data["u"] = env.u_n().x.array.real
        u_grid.set_active_scalars("u")

        w_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
        w_grid.point_data["w"] = env.w_n().x.array.real
        w_grid.set_active_scalars("w")

        u_plotter = pyvista.Plotter()
        u_plotter.add_mesh(u_grid, show_edges=False, clim=[0, 1])
        u_plotter.view_xy()
        u_plotter.show(auto_close=False, interactive_update=True)

        w_plotter = pyvista.Plotter()
        w_plotter.add_mesh(w_grid, show_edges=False, clim=[0, 1])
        w_plotter.view_xy()
        w_plotter.show(auto_close=False, interactive_update=True)

        for i in range(5000):
            if i == 30:
                env.intensity = 0
                env.compute_right_hand_side()

            env.advance_time(0.5)
            # sleep(0.1)
            print(i, flush=True)

            # Plot the transmembrane potential
            u_grid.point_data["u"] = env.u_n().x.array.real
            w_grid.point_data["w"] = env.w_n().x.array.real
            u_plotter.update()
            w_plotter.update()
