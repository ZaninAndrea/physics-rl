from petsc4py import PETSc
from dolfinx import fem, mesh
import ufl
from petsc4py import PETSc
from dolfinx import fem, mesh
from typing import Any, Optional, List, Union, Callable


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
    def set_A(self, a, bcs: List[fem.DirichletBCMetaClass]):
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
