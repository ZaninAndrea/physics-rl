import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="physicsrl",
    version="1.0.0",
    author="Andrea Zanin",
    description="A package for reinforcement learning in physics simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    py_modules=[
        "physicsrl",
    ],
    package_dir={"": "."},
    install_requires=[
        "tf-agents==0.17.0",
        "tensorflow==2.13.0",
        "numpy==1.24.0",
        "imageio==2.4.0",
        "jupyterlab",
        "matplotlib",
        "pyglet",
        "pyvirtualdisplay",
    ],
)
