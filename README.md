# Physics-rl

Physics-rl allows users to easily train tensorflow agents using reinforcement learning in environments simulated using FENiCSx.

This library provides the following functionalities:

-   Tools to handle a mix of distributed computation and single-node computation (Coordinator and Coordinated classes and `@parallel` decorator).
-   Ready to use simulation of two physical environments governed by the heat diffusion equation (HeatEnvironment class) and by a monodomain Mitchell-Schaeffer model (MonodomainMitchellSchaeffer class).
-   A class that handles data collection and training using user-provided agents and environments with reasonable opinionated defaults (Dojo class).

A detailed description of the design of the library is available in the `PACS Report.pdf` file.

## Coordination system

Any class that needs to run distributed FENiCSx computation should inherit from Coordinated and the methods that need to be run on all processes should be marked with the `@parallel` decorator. The result is a class that can be used in the leader process as if it was running in a single process (thus allowing compatibility with tf-agent's ecosystem), while still running on all processes the necessary computations.

All processes should have a Coordinator instance; in most cases there should be exactly one per process. All instances inheriting from Coordinated must be registered in the Coordinator, the corresponding instances of the same class in different processes are registered in a common namespace (the namespace can be manually set, but in most cases it is inferred automatically). This allows a function call in an instance in the leader process to be replicated on the correct instances in the follower processes.

The typical usage of the library is as follows

```python
from physicsrl import coordinated

coordinator = coordinated.Coordinator()
env = System(coordinator)     # System inherits from coordinated.Coordinated and PyEnvironment
with coordinator:
    if coordinator.is_leader():
        env = tf_py_environment.TFPyEnvironment(env)
        q_net = QNetwork(env.observation_spec(), env.action_spec())

        dojo = Dojo(q_net, env)

        dojo.train(100)
```

Check out the file `examples/example_dojo_fenics.py` for a complete example of how the library can be used.

## Setup

To install just the `physicsrl` library run

```
pip install physicsrl
```

To install `physicsrl` and the libraries for FENiCSx and MPI follow these steps:

-   Uninstall MPI if you have already installed it (in some cases it conflicts with conda's MPI installation)
-   Create environment

```
conda create --name physicsrl python==3.10
conda activate physicsrl
```

-   install dolfinx following the [official instructions](https://github.com/FEniCS/dolfinx/blob/main/README.md#installation)

```
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

-   install `physicsrl`

```
pip install physicsrl
```

## Compatibility

Python 3.9 or more recent is required.
