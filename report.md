# Introduction

The project has been developed by Andrea Zanin with the supervision of professor Stefano Pagani.

The goal of this project was developing a library to simplify the development and training of reinforcement learning agents interacting with a simulated physical environment. We identified tensorflow's tf-agents library as the most widely used library in the reinforcement learning field; as far as physical environments are concerned we chose FENiCSx as library to perform the numerical simulations.

# Problem

The numerical simulations can be very computationally intensive and thus need to be executed in a distributed environment; FENiCSx achieves this using MPI to handle the low-level coordination of the processes.  
The tf-agents library on the other hand is not designed to work with MPI, this means that using it in conjunction with FENiCSx is not straightforward. Two key problems arise if we run tf-agent's code on all processes and use MPI to distribute the FENiCSx computation:

-   tf-agent's internals and many of the libraries in its ecosystem have non-deterministic behaviour (e.g. usage of random number generators), to guarantee correctness we must ensure that this behaviour is identical across all processes. Doing this is often non-trivial or not at all possible, e.g. if the library does not allow seeding the RNG.
-   the training of the agent is computationally expensive, so it is wasteful to do it on every process independently.

On top of solving those problems, we wanted to reduce the amount of boilerplate code needed to handle data collection and training of the agent, thus allowing faster iteration on research ideas.

# Solution

We designed a system that ensures all of tf-agent's computation is done on a single process (the leader process) and the results of the computation are then replicated across all processes. This approach ensures that non-deterministic behaviour is identical in all processes and that no redundant computation is performed.

The system can be extended to allow distributing tf-agent's computation across several processes, with the only limitation that the only process (or thread if using python async capabilities) that runs both tf-agent's computation and FENiCSx computation is the leader process. We deem this to be an acceptable limitation, because tf-agent runs mostly on GPU while FENiCSx runs on CPU, so it is reasonable to use different machines for the two tasks.

The classes of the physics-rl library have been structured to allow developers to use just a subset of the functionalities without necessarily using all the library. In particular the functionalities are:

-   tools to handle a mix of distributed computation and single-node computation (Coordinator and Coordinated classes and @parallel decorator)
-   ready to use simulation of two physical environments governed by the heat diffusion equation (HeatEnvironment class) and by a monodomain Mitchell-Schaeffer model (MonodomainMitchellSchaeffer class). These environments use Coordinated to distribute computation.
-   a class that handles data collection and training using user-provided agents and environments with reasonable opinionated defaults (Dojo class)

# Coordination system

Any class that needs to run distributed FENiCSx computation should inherit from Coordinated and the methods that need to be run on all processes should be marked with the `@parallel` decorator. The resulting behaviour is a class that can be used in the leader process as if it was running in a single process (thus allowing compatibility with tf-agent's ecosystem), while still running on all processes the necessary computations.

All processes should have a Coordinator instance; in most cases there should be exactly one per process, but in some cases more than one may be necessary (see Coordinator instead of multiple communicators).
All instances inheriting from Coordinated must be registered in the Coordinator, the corresponding instances of the same class in different processes are registered in a common namespace (the namespace can be manually set, but in most cases it is inferred automatically). This allows a function call in an instance in the leader process to be replicated on the correct instances in the follower processes.

## Method call replication

The system that guarantees that a method call on the leader instance is replicated on all follower instances is the following:

1. The @parallel decorator wraps the method implementation so that when the method is called the Coordinator is also notified (deriving from Coordinated is necessary so that the decorator can get access to the Coordinator instance).
2. When the leader Coordinator is notified of the method call it broadcasts the method name, the method arguments and the instance namespace to the other Coordinators using an MPI Broadcast
3. When a follower Coordinator receives the method call broadcast it calls the method with the received name on the instance in the received namespace.

When a method A decorated with @parallel calls another method B also decorated with @parallel the algorithm stated above would lead to duplicate calls to B on follower processes: the method is called both by method A running in the follower process and by the broadcasted B call coming from the leader process. For this reason the Coordinator class tracks whether a parallel method is being executed and in that case it does not broadcast nested calls.

The follower processes should be waiting for broadcast messages coming from the leader process, furthermore they should shutdown when the leader process terminates; this is all handled using the `with coordinator` construct.

## Coordinator instead of multiple communicators

Instead of using the Coordinator class to distinguish messages in different namespaces we could have used a different communicator for each namespace (e.g. running MPI_Comm_dup in the constructor of Coordinated), but then the follower replicas would need to be multithreaded if several Coordinated instances are needed, this would have made the library more complex for the end user since multithreading is still a relatively niche python feature.

We decided to introduce the Coordinator class that multiplexes the messages of all namespaces in a single communicator, this has the advantage that no multithreading is needed to support several Coordinated instances. If a user of the library needs the Coordinated classes to operate in parallel, they can still use multithreading and different communicators: they just need to instantiate multiple Coordinator classes.

## Python implementation choices

To aid users of the library and guarantee correctness of our implementation all functions in the library use the python type hint feature.

The Coordinated class in a `register_coordinator` method instead of the constructor to simplify the implementation of classes that use multiple inheritance; indeed we expect most classes that derive from Coordinated to also derive from tf-agent's PyEnvironment.

# A builder for FENiCSx

In the context of training an agent with reinforcement learning we need to iteratively update the PDE that we want to solve in response to the agent's actions. To allow user's of the library to do this easily we implemented the TimeProblem class, which follows the builder pattern: it allows the user to set the bilinear form, the right hand side, the boundary conditions, the initial condition and the time step of the problem, as well as overwrite any of them at any point in the simulation.

The TimeProblem is designed to handle problems that have been discretized in time with a finite differences scheme and then the semidiscretized problem has been expressed in weak formulation, so that it can be solved by FENiCSx.

## Heat equation example

To showcase how the library can be used we implemented an environment governed by the heat equation and an agent that can observe some metrics derived from the environment (the temperature in the 4 quadrants). The agent can stop the simulation at any time, it receive a penalty for waiting and at the end gets a reward proportional to the average temperature.

Thanks to our library we have implemented the numerical simulation and the agent training in less than 300 lines of code.

## Monodomain Mitchell-Schaeffer simulation

To demonstrate the flexibility of the library we also implemented a simulation of the monodomain Mitchell-Schaeffer equation, which models the trans-membrane potential in the heart. In particular we reproduced the typical spiral shape of reentrant arrhythmias.

The PDE from which we started is the following:
