from dolfinx import plot
import pyvista
from coordinated import (
    Coordinator,
)
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from fem import MonodomainMitchellSchaeffer
from math import floor


def half_plane(x):
    v = np.array(0.0 * x[0])
    v[x[0] < 0] = np.minimum(-x[0][x[0] < 0] / 20, np.ones_like(x[0][x[0] < 0]))

    return v


nx, ny = 300, 300
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
    lambda x: np.zeros_like(x[0]),
    half_plane,
    dt=0.03,
)

show_plot = True
with coordinator:
    if coordinator.is_leader():
        if show_plot:
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
            u_plotter.window_size = 2000, 2000

            w_plotter = pyvista.Plotter()
            w_plotter.add_mesh(w_grid, show_edges=False, clim=[0, 1])
            w_plotter.view_xy()
            w_plotter.show(auto_close=False, interactive_update=True)

        def I_app(x):
            v = np.array(0.0 * x[0])
            v[np.logical_and(x[1] > 0, np.abs(x[0]) < 5)] = 50

            return v

        env.set_I_app(I_app)

        for i in range(2000):
            if i == 30:

                def I_app(x):
                    return x[0] * 0.0

                env.set_I_app(I_app)

            env.advance_time(0.5)
            print(i, flush=True)

            # # Plot the transmembrane potential
            if show_plot:
                u_grid.point_data["u"] = env.u_n().x.array.real
                w_grid.point_data["w"] = env.w_n().x.array.real
                time = floor(i * 0.5)
                u_plotter.add_title(f"t = {time}")
                u_plotter.update()
                w_plotter.update()
                u_plotter.screenshot(f"./screenshots/{i:05d}.png")
