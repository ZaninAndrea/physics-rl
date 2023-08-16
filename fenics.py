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


class HeatEnvironment(
    py_environment.PyEnvironment,
    Coordinated,
    ABC,
):
    def __init__(
        self,
        coordinator: Coordinator,
        domain,
        dirichlet_bc,
        initial_condition,
        source_term,
        dt=0.01,
        function_space=None,
    ):
        self.register_coordinator(coordinator)

        self.dt = dt
        self.domain = domain

        V = function_space
        if function_space is None:
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

    def problem(self):
        return self._problem

    def u_n(self):
        return self._u_n

    def set_initial_condition(self, initial_condition):
        self._problem.set_initial_condition(initial_condition)

    def set_source_term(self, source_term):
        L = (self.u_n() + self.dt * source_term) * self._v * ufl.dx
        self._problem.set_L(L)

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

    @parallel
    def compute_ufl_form(self, form):
        return self.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(form)),
            op=MPI.SUM,
        )

    @parallel
    def advance_time(self, T):
        self._problem.solve(T)

    # handle_reset can be overridden to implement custom reset behavior.
    def handle_reset(self):
        pass

    @parallel
    def _reset(self):
        self._problem.reset()
        self.handle_reset()

        return ts.restart(self.get_observation())

    @abstractmethod
    def get_observation(self):
        pass


class TimeProblem:
    def __init__(self, domain):
        # Create linear solver
        self.domain = domain
        self._solver = PETSc.KSP().create(domain.comm)
        self._solver.setType(PETSc.KSP.Type.PREONLY)
        self._solver.getPC().setType(PETSc.PC.Type.LU)

        self.u_n = None
        self._is_running = False

    def set_A(self, a, bcs):
        self._bcs = bcs

        self._V = a.arguments()[-1].ufl_function_space()
        self._bilinear_form = fem.form(a)
        self._A = fem.petsc.assemble_matrix(self._bilinear_form, bcs=bcs)
        self._A.assemble()
        self._solver.setOperators(self._A)

    def set_L(self, L):
        self._linear_form = fem.form(L)
        self._b = fem.petsc.create_vector(self._linear_form)

    def set_dt(self, dt):
        self.dt = dt

    def set_initial_condition(self, initial_condition):
        self.initial_condition = initial_condition
        # if not self._is_running and self.u_n != None:
        #     self.u_n.interpolate(self.initial_condition)

    def set_u_n(self, u_n):
        self.u_n = u_n
        if not self._is_running and self.initial_condition != None:
            self.u_n.interpolate(self.initial_condition)

    def _setup_pyvista(
        self,
        uh,
        gif_path,
        clim=None,
        cmap=plt.cm.get_cmap("viridis", 25),
        scalar_bar_args=dict(
            title_font_size=25,
            label_font_size=20,
            fmt="%.2e",
            color="black",
            position_x=0.1,
            position_y=0.8,
            width=0.8,
            height=0.1,
        ),
    ):
        pyvista.start_xvfb()

        grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(self._V))

        plotter = pyvista.Plotter()
        plotter.open_gif(gif_path)

        grid.point_data["uh"] = uh.x.array
        warped = grid.warp_by_scalar("uh", factor=1)

        if clim is None:
            clim = [min(uh.x.array), max(uh.x.array)]

        renderer = plotter.add_mesh(
            warped,
            show_edges=True,
            lighting=False,
            cmap=cmap,
            scalar_bar_args=scalar_bar_args,
            clim=clim,
        )

        return (plotter, grid, renderer)

    def reset(self):
        self._is_running = False

    def solve(self, T, xdmf_output_path=None, gif_path=None):
        # Setup solution function
        uh = fem.Function(self._V)
        uh.name = "uh"

        if not self._is_running:
            uh.interpolate(self.initial_condition)
            self._is_running = True

        t = 0

        if gif_path != None:
            plotter, grid, _ = self._setup_pyvista(
                uh,
                gif_path,
            )

        if xdmf_output_path != None:
            xdmf = io.XDMFFile(self.domain.comm, xdmf_output_path, "w")
            xdmf.write_mesh(self.domain)
            xdmf.write_function(uh, t)

        # Time loop
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

            # Write solution to file
            if xdmf_output_path != None:
                xdmf.write_function(uh, t)

            # Update plot
            if gif_path != None:
                warped = grid.warp_by_scalar("uh", factor=1)
                plotter.update_coordinates(warped.points.copy(), render=False)
                plotter.update_scalars(uh.x.array, render=False)
                plotter.write_frame()

        if gif_path != None:
            plotter.close()

        if xdmf_output_path != None:
            xdmf.close()

        return uh
