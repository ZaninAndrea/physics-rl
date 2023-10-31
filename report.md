# Introduction

The project has been developed by Andrea Zanin with the supervision of professor Stefano Pagani.

The goal of this project was developing a library to simplify the development and training of reinforcement learning agents interacting with a simulated physical environment. We identified tensorflow's tf-agents library as the most widely used library in the reinforcement learning field; as far as physical environments are concerned we chose FENiCS as library to perform the numerical simulations.

# Problem

The numerical simulations can be very computationally intensive and thus need to be executed in a distributed environment; FENiCS achieves this using MPI to handle the low-level coordination of the processes.  
The tf-agents library on the other hand is not designed to work with MPI, this means that using it in conjunction with FENiCS is not straightforward. Two key problems arise if we run tf-agent's code on all processes and use MPI to distribute the FENiCS computation:

-   tf-agent's internals and many of the libraries in its ecosystem have non-deterministic behaviour (e.g. usage of random number generators), to guarantee correctness we must ensure that this behaviour is identical across all processes. Doing this is often non-trivial or not at all possible, e.g. if the library does not allow seeding the RNG.
-   the training of the agent is computationally expensive, so it is very wasteful to do it on every process independently.

We also wanted to reduce the amount of boilerplate code needed to handle data collection and training of the agent, thus allowing faster iteration on research ideas.

# Solution

We designed a system that ensures all of tf-agent's computation is done on a single process (the leader process) and the results of the computation are then replicated across all processes. This approach ensures that non-deterministic behaviour is identical in all processes and that no redundant computation is performed.

The system can be extended to allow distributing tf-agent's computation across several processes, with the only limitation that the only process that runs both tf-agent's computation and FENiCS computation is the leader process. We deem this to be an acceptable limitation, because tf-agent runs mostly on GPU while FENiCS runs on CPU, so it is reasonable to use different machines for the two tasks.

The classes of the physics-rl library have been structured to allow developers to use just a subset of the functionalities without necessarily using all the library. In particular the functionalities are:

-   tools to handle a mix of distributed computation and single-node computation (Coordinator and Coordinated classes and @parallel decorator)
-   ready to use simulation of two physical environments governed by the heat diffusion equation (HeatEnvironment class) and by a monodomain Mitchell-Schaeffer model (MonodomainMitchellSchaeffer class). These environments use Coordinated to distribute computation.
-   a class that handles data collection and training using user-provided agents and environments with reasonable opinionated defaults (Dojo class)

# Coordination system

Any class that needs to run distributed FENiCS computation should inherit from Coordinated and the methods that need to be run on all processes should be marked with the `@parallel` decorator. The resulting behaviour is a class that can be used in the leader process as if it was running in a single process (thus allowing compatibility with tf-agent's ecosystem), while still running on all processes the necessary computations.

All processes should have a Coordinator instance; in most cases there should be exactly one per process, but in some cases more than one may be necessary (see Appendix 1).
All instances inheriting from Coordinated must be registered in the Coordinator, the corresponding instances of the same class in different processes are registered under a common namespace.

Our system to manage distributed and non-distributed computation is based on the Coordinator class.
Objects that run distributed computations must be registered in a Coordinator under a namespace, each namespace is connected to one instance in each process.

# Appendix

## Coordinator instead of multiple communicators

Instead of using the Coordinator class to distinguish messages in different namespaces we could have used a different communicator for each namespace (e.g. running MPI_Comm_dup in the constructor of Coordinated), but then the follower replicas would need to be multithreaded if several Coordinated instances are needed, this would have made the library more complex for the end user.

We decided to introduce the Coordinator class that multiplexes the messages of all namespaces in a single communicator, this has the advantage that no multithreading is needed to support several Coordinated instances. If a user of the library needs the Coordinated classes to operate in parallel, they can still use multithreading and different communicators: they just need to instantiate multiple Coordinator classes.
