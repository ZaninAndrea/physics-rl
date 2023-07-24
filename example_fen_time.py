from fenics import TimeProblem
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh

# Define mesh
nx, ny = 50, 50
dt = 0.01
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
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
    PETSc.ScalarType(1), fem.locate_dofs_topological(V, fdim, boundary_facets), V
)

# Define variational problem
u_n = fem.Function(V)
u_n.name = "u_n"

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

f = fem.Constant(domain, PETSc.ScalarType(0))
L = (u_n + dt * f) * v * ufl.dx

# Solve using TimeProblem
problem = TimeProblem(domain)
problem.set_A(a, [bc])
problem.set_L(L)
problem.set_dt(dt)
problem.set_initial_condition(lambda x: np.array(0.0 * x[0]))
problem.set_u_n(u_n)

uh = problem.solve(1, gif_path="fen_time.gif")

# f = fem.Constant(domain, PETSc.ScalarType(-0.5))
# L = (u_n + dt * f) * v * ufl.dx
# problem.set_L(L)

# problem.solve(0.7, gif_path="fen_time.gif")

# problem.reset()
# uh = problem.solve(1, gif_path="fen_time.gif")

# # Compute total heat
heat_local = fem.assemble_scalar(fem.form(uh * ufl.dx))
heat = domain.comm.allreduce(heat_local, op=MPI.SUM)
print("Total heat: ", heat)
