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


DT = 0.03
RESOLUTION = 3000


def half_plane(x):
    v = np.array(0.0 * x[0])
    v[x[0] < 0] = np.minimum(-x[0][x[0] < 0] / 20, np.ones_like(x[0][x[0] < 0]))

    return v


# zero is the zero function
def zero(x):
    return np.zeros_like(x[0])


# Create a system with the PyEnvironment methods stubbed out
# so that we can use test the performance of the system
# in isolation.
class System(MonodomainMitchellSchaeffer):
    def __init__(self, coordinator: Coordinator):
        self.comm = coordinator.MPI_Comm()

        # Setup equation parameters
        nx, ny = RESOLUTION, RESOLUTION
        domain = mesh.create_rectangle(
            self.comm,
            [np.array([-50, -50]), np.array([50, 50])],
            [nx, ny],
            mesh.CellType.quadrilateral,
        )

        super().__init__(
            coordinator,
            domain,
            zero,
            half_plane,
            dt=DT,
        )

    def get_observation(self):
        pass

    def action_spec(self):
        pass

    def observation_spec(self):
        pass

    def _reset(self, seed=None):
        pass

    def _step(self, action):
        pass


# Create the environment and the agent, then
# train the agent on the environment
coordinator = Coordinator()
env = System(coordinator)


def active_stimulus(x):
    v = np.array(0.0 * x[0])
    v[np.logical_and(x[1] > 0, np.abs(x[0]) < 5)] = 50

    return v


with coordinator:
    if coordinator.is_leader():
        # Enable the stimulus
        env.set_I_app(active_stimulus)

        for i in range(100):
            print(f"Step {i}", flush=True)

            # At step 30, turn off the stimulus
            if i == 30:
                env.set_I_app(zero)

            env.advance_time(DT)
