# Setup

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

-   install the other dependencies

```
pip install -r requirements.txt
```

# Compatibility

Python 3.9 or more recent is required
