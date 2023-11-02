from dolfinx import fem, mesh
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from coordinated import (
    Coordinator,
    Coordinated,
    parallel,
)
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Callable
from fenics import TimeProblem


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
