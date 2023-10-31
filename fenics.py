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


# TimeProblem wraps a fenics variational problem and allows the PDE
# to be updated during the simulation.
class TimeProblem:
    def __init__(self, domain):
        # Create fenics solver
        self.domain = domain
        self._solver = PETSc.KSP().create(domain.comm)
        self._solver.setType(PETSc.KSP.Type.PREONLY)
        self._solver.getPC().setType(PETSc.PC.Type.LU)

        self.u_n = None
        self._is_running = False

    # Set the bilinear form of the problem
    def set_A(self, a, bcs):
        self._bcs = bcs

        self._V = a.arguments()[-1].ufl_function_space()
        self._bilinear_form = fem.form(a)
        self._A = fem.petsc.assemble_matrix(self._bilinear_form, bcs=bcs)
        self._A.assemble()
        self._solver.setOperators(self._A)

    # Set the right hand side of the problem
    def set_L(self, L):
        self._linear_form = fem.form(L)
        self._b = fem.petsc.create_vector(self._linear_form)

    # Set the time step of the simulation
    def set_dt(self, dt):
        self.dt = dt

    # Set the initial condition of the problem
    def set_initial_condition(self, initial_condition):
        self.initial_condition = initial_condition

    # Set the solution at the current time step
    def set_u_n(self, u_n):
        self.u_n = u_n
        if not self._is_running and self.initial_condition != None:
            self.u_n.interpolate(self.initial_condition)

    # Reset the problem so that it will start from the initial condition again
    def reset(self):
        self._is_running = False

    # Solve the problem for T time units
    def solve(self, T):
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
        domain: Any,
        dirichlet_bc: Any,
        initial_condition: Any,
        source_term: Any,
        dt: float = 0.01,
        V: Any = None,
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
    def u_n(self) -> Any:
        return self._u_n

    # Change the initial condition of the problem
    def set_initial_condition(self, initial_condition: Any) -> None:
        self._problem.set_initial_condition(initial_condition)

    # Change the source term of the problem
    def set_source_term(self, source_term: Any):
        L = (self.u_n() + self.dt * source_term) * self._v * ufl.dx
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
            + self.dt * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
        )

        self._problem.set_A(a, [bc])

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
