import pyvista
from dolfinx import plot
from petsc4py import PETSc
from dolfinx import fem, mesh
from typing import Any, Optional, List, Union, Callable
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from coordinated import Coordinator, Coordinated, parallel
from abc import ABC, abstractmethod
import numpy as np
import ufl
from mpi4py import MPI
from typing import Any, Union, Callable, Optional
from ufl import conditional
import matplotlib.pyplot as plt
import ufl


# TimeProblem wraps a fenics variational problem and allows the PDE
# to be updated during the simulation.
class TimeProblem:
    def __init__(self, domain: mesh.Mesh):
        # Create fenics solver
        self.domain = domain
        self._solver = PETSc.KSP().create(domain.comm)
        self._solver.setType(PETSc.KSP.Type.PREONLY)
        self._solver.getPC().setType(PETSc.PC.Type.LU)

        self.u_n: Optional[fem.Function] = None
        self._is_running = False

    # Set the bilinear form of the problem
    def set_A(self, a: ufl.Form, bcs: List[fem.DirichletBCMetaClass]):
        self._bcs = bcs

        self._V: ufl.FunctionSpace = a.arguments()[-1].ufl_function_space()
        self._bilinear_form = fem.form(a)
        self._A = fem.petsc.assemble_matrix(self._bilinear_form, bcs=bcs)
        self._A.assemble()
        self._solver.setOperators(self._A)

    # Set the right hand side of the problem
    def set_L(self, L: ufl.Form):
        self._linear_form = fem.form(L)
        self._b = fem.petsc.create_vector(self._linear_form)

    # Set the time step of the simulation
    def set_dt(self, dt: float):
        self.dt = dt

    # Set the initial condition of the problem
    def set_initial_condition(
        self, initial_condition: Union[Callable, fem.Expression, fem.Function]
    ):
        self.initial_condition = initial_condition

    # Set the solution at the current time step
    def set_u_n(self, u_n: fem.Function):
        self.u_n = u_n
        if not self._is_running and self.initial_condition != None:
            self.u_n.interpolate(self.initial_condition)

    # Reset the problem so that it will start from the initial condition again
    def reset(self):
        self._is_running = False
        self.u_n.interpolate(self.initial_condition)

    # Solve the problem for T time units
    def solve(self, T: float):
        # Setup solution function
        uh = fem.Function(self._V)
        uh.name = "uh"

        if not self._is_running:
            uh.interpolate(self.initial_condition)
            self._is_running = True

        # Time loop
        t = 0
        while t < T:
            t += self.dt

            # Update the right hand side reusing the initial vector
            with self._b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self._b, self._linear_form)

            # Apply Dirichlet boundary condition to the vector
            fem.petsc.apply_lifting(self._b, [self._bilinear_form], [self._bcs])
            self._b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            fem.petsc.set_bc(self._b, self._bcs)

            # Solve linear problem
            self._solver.solve(self._b, uh.vector)
            uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            if self.u_n != None:
                self.u_n.x.array[:] = uh.x.array

        return uh


# HeatEnvironment is an abstract base class that implements a
# heat equation environment for reinforcement learning.
# The PDE defining the environment and the solution can be modified
# during the simulation to account for interactions with the agent.
class HeatEnvironment(
    py_environment.PyEnvironment,
    Coordinated,
    ABC,
):
    def __init__(
        self,
        coordinator: Coordinator,
        domain: mesh.Mesh,
        dirichlet_bc: List[fem.DirichletBCMetaClass],
        initial_condition: Union[Callable, fem.Expression, fem.Function],
        source_term: Union[fem.Expression, fem.Function],
        dt: float = 0.01,
        V: Optional[fem.FunctionSpace] = None,
    ) -> None:
        self.register_coordinator(coordinator)

        self.dt = dt
        self.domain = domain

        if V is None:
            V = fem.FunctionSpace(domain, ("CG", 1))
        self._V = V

        # Define variational problem
        self._u, self._v, self._u_n = (
            ufl.TrialFunction(V),
            ufl.TestFunction(V),
            fem.Function(V),
        )
        self._u_n.name = "u_n"
        self._problem = TimeProblem(domain)

        self.set_source_term(source_term)
        self.set_dirichlet_bc(dirichlet_bc)
        self.set_initial_condition(initial_condition)
        self._problem.set_dt(dt)
        self._problem.set_u_n(self._u_n)

    # Get the current problem
    def problem(self) -> TimeProblem:
        return self._problem

    # Get the current solution
    def u_n(self) -> fem.Function:
        return self._u_n

    # Change the initial condition of the problem
    def set_initial_condition(
        self, initial_condition: Union[Callable, fem.Expression, fem.Function]
    ) -> None:
        self._problem.set_initial_condition(initial_condition)

    # Change the source term of the problem
    def set_source_term(self, source_term: Union[fem.Expression, fem.Function]):
        L = (self.u_n() + self.dt * source_term) * self._v * ufl.dx
        self._problem.set_L(L)

    # Change the Dirichlet boundary condition of the problem
    def set_dirichlet_bc(self, dirichlet_bc: List[fem.DirichletBCMetaClass]):
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
            + self.dt * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
        )

        self._problem.set_A(a, [bc])

    # Compute the integral of a UFL form over the domain in parallel
    @parallel
    def compute_ufl_form(self, form: ufl.Form):
        return self.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(form)),
            op=MPI.SUM,
        )

    # Advance the time by T in the problem
    @parallel
    def advance_time(self, T: float) -> None:
        self._problem.solve(T)

    # handle_reset can be overridden to implement custom reset behavior,
    # it will be called after the problem is reset, but before
    # the observation is computed.
    def handle_reset(self):
        pass

    # Reset the environment, this method is called automatically be tf-agents
    @parallel
    def _reset(self):
        self._problem.reset()
        self.handle_reset()

        return ts.restart(self.get_observation())

    # get_observation should be overridden to implement custom observations
    @abstractmethod
    def get_observation(self) -> Any:
        pass


# MonodomainMitchellSchaeffer implements a monodomain Mitchell-Schaeffer environment.
# The PDE defining the environment and the solution can be modified
# during the simulation to account for interactions with the agent.
class MonodomainMitchellSchaeffer(
    py_environment.PyEnvironment,
    Coordinated,
    ABC,
):
    def __init__(
        self,
        coordinator: Coordinator,
        domain: mesh.Mesh,
        initial_condition_u: Union[Callable, fem.Expression, fem.Function],
        initial_condition_w: Union[Callable, fem.Expression, fem.Function],
        I_app: Union[Callable, fem.Expression, fem.Function] = lambda x: x[0] * 0.0,
        tau_in: float = 0.3,
        tau_out: float = 6.0,
        tau_open: float = 75.0,
        tau_close: float = 80.0,
        u_gate: float = 0.13,
        D: float = 0.013,
        dt: float = 0.01,
        V: Optional[fem.FunctionSpace] = None,
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

        self._I_app.interpolate(I_app)
        self.compute_right_hand_side()
        self.set_A()

        self.set_initial_condition(initial_condition_u, initial_condition_w)
        self._problem.set_dt(dt)
        self._problem.set_u_n(self._u_n)

    # Get the current problem
    def problem(self) -> TimeProblem:
        return self._problem

    # Get the current transmembrane potential
    def u_n(self) -> fem.Function:
        return self._u_n

    # Get the current gate variable
    def w_n(self) -> fem.Function:
        return self._w_n

    # Reset the gate variable
    def reset_w_n(self) -> None:
        self._w_n.interpolate(self._initial_condition_w)
        self.compute_right_hand_side()

    # Change the initial condition of the problem
    def set_initial_condition(
        self,
        initial_u: Union[Callable, fem.Expression, fem.Function],
        initial_w: Union[Callable, fem.Expression, fem.Function],
    ) -> None:
        self._problem.set_initial_condition(initial_u)

        self._initial_condition_w = initial_w
        self._w_n.interpolate(initial_w)

    @parallel
    def set_I_app(self, I_app: Union[Callable, fem.Expression, fem.Function]) -> None:
        self._I_app.interpolate(I_app)
        self.compute_right_hand_side()

    def compute_right_hand_side(self):
        J_in = self._w_n * self._u_n * self._u_n * (1 - self._u_n) / self._tau_in
        J_out = -self._u_n / self._tau_out
        L = (self._u_n + self.dt * (J_in + J_out + self._I_app)) * self._v * ufl.dx

        self._problem.set_L(L)

    def set_A(self):
        # Define variational problem
        a = (
            self._u * self._v * ufl.dx
            + self.dt * self._D * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
        )

        self._problem.set_A(a, [])

    # Compute the integral of a UFL form over the domain in parallel
    @parallel
    def compute_ufl_form(self, form: ufl.Form):
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

    # get_observation should be overridden to implement custom observations
    @abstractmethod
    def get_observation(self) -> Any:
        pass

    def save_u_image(self, image_path: str):
        if self.domain.comm.rank == 0:
            print("Saving screenshot", flush=True)
            u_topology, u_cell_types, u_geometry = plot._(self._V)

            u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
            u_grid.point_data["u"] = self.u_n().x.array.real
            u_grid.set_active_scalars("u")

            u_plotter = pyvista.Plotter(off_screen=True)
            u_plotter.add_mesh(u_grid, show_edges=False, clim=[0, 1])
            u_plotter.view_xy()
            u_plotter.window_size = 2000, 2000

            u_plotter.screenshot(image_path)
            u_plotter.close()
