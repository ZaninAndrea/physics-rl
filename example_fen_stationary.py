from dolfinx import mesh
from mpi4py import MPI
from dolfinx.fem import FunctionSpace, Function
from dolfinx import fem
from dolfinx import plot
import numpy
import pyvista
from petsc4py.PETSc import ScalarType
import ufl

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
V = FunctionSpace(domain, ("CG", 1))


def get_problem(V):
    # Boundary conditions
    uD = Function(V)
    uD.interpolate(lambda x: 5 + (x[0] - 0.5) ** 2 + 2 * (x[1] - 0.5) ** 2)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    # Defining trial and test function
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Defining the source term
    f = fem.Constant(domain, ScalarType(-6))

    # Defining the variational problem
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )

    return problem


problem = get_problem(V)
uh = problem.solve()


# Computing the error
def compute_error(uh, V):
    V2 = fem.FunctionSpace(domain, ("CG", 2))
    uex = fem.Function(V2)
    uex.interpolate(lambda x: 5 + (x[0] - 0.5) ** 2 + 2 * (x[1] - 0.5) ** 2)

    L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    uD = Function(V)
    uD.interpolate(lambda x: 5 + (x[0] - 0.5) ** 2 + 2 * (x[1] - 0.5) ** 2)
    error_max = numpy.max(numpy.abs(uD.x.array - uh.x.array))

    return (error_L2, error_max)


if domain.comm.rank == 0:
    error_L2, error_max = compute_error(uh, V)
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")


# Plot solutions
def plot_solution(uh, V):
    pyvista.start_xvfb()

    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show()


if domain.comm.rank == 0:
    plot_solution(uh, V)
