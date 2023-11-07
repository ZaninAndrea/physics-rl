from mpi4py import MPI
import random as rand
from collections.abc import Callable
from typing import Any, Optional


# Generate a random number between 0 and 1 ensuring
# that all nodes read the same value
def random(comm: MPI.Comm) -> float:
    return comm.bcast(rand.random(), root=0)


# Generate a random integer between a and b ensuring
# that all nodes read the same value
def randint(comm: MPI.Comm, a: int, b: int) -> int:
    return comm.bcast(rand.randint(a, b), root=0)


# Coordinator class that manages the registration of objects
# and the broadcasting of commands to execute methods on
# registered objects.
class Coordinator(object):
    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD) -> None:
        super(Coordinator, self).__init__()

        self.comm = comm
        self.objects = {}
        self.is_leader_inside_with_statement = False
        self.registrationCounter = 0

    def MPI_Comm(self) -> MPI.Comm:
        return self.comm

    def is_leader(self) -> bool:
        return self.comm.rank == 0

    # Register an object under a given namespace; if the namespace is not
    # provided, an incremental counter is used as namespace.
    # Unless you have different register calls on different nodes, the
    # default namespace is usually fine.
    def register(self, object: Any, namespace: Optional[str] = None) -> str:
        if namespace is None:
            namespace = "object{}".format(self.registrationCounter)

        self.objects[namespace] = object
        self.registrationCounter += 1

        return namespace

    # Broadcast a command to all nodes, the command will be executed by the
    # object registered under the given namespace on each node.
    def broadcast(self, namespace: str, method: str, *args: Any, **kwargs: Any) -> None:
        if not self.is_leader_inside_with_statement:
            raise Exception(
                "Broadcast can only be called by the leader node inside a `with coordinator` block"
            )

        self.comm.bcast(
            {
                "namespace": namespace,
                "method": method,
                "args": args,
                "kwargs": kwargs,
            },
            root=0,
        )

    # Context manager that ensures that while the leader node is executing
    # the code inside the with statement, the other nodes are waiting for
    # commands to be broadcasted.
    # After the with statement is exited by the leader, the other nodes
    # will enter the with statement, the user should ensure that no
    # parallel code is executed by the follower nodes.
    def __enter__(self) -> None:
        self.is_leader_inside_with_statement = True

        if self.comm.rank != 0:
            while True:
                command = self.comm.bcast(None, root=0)
                if command == "stop":
                    self.is_leader_inside_with_statement = False
                    break
                else:
                    object = self.objects[command["namespace"]]
                    if object is None:
                        raise Exception(
                            "Object with namespace {} not found".format(
                                command["namespace"]
                            )
                        )

                    method = getattr(object, command["method"])
                    method(*command["args"], **command["kwargs"])

            return None

    def __exit__(self, *args: Any) -> None:
        if self.comm.rank == 0:
            self.comm.bcast("stop", root=0)
            self.is_leader_inside_with_statement = False

    def is_ready_to_broadcast(self) -> bool:
        return self.is_leader_inside_with_statement


# Base class that connects to a Coordinator and provides
# a utilitity to execute methods in parallel across all
# instances of the class managed by the Coordinator.
class Coordinated(object):
    def register_coordinator(
        self, coordinator: Coordinator, namespace: Optional[str] = None
    ) -> None:
        # The registration is not performed in the constructor
        # to simplify usage in the case of multiple inheritance
        self.__coordinator = coordinator
        self.__namespace = coordinator.register(self, namespace=namespace)
        self.__is_inside_parallel_block = False

    def run_method_in_parallel(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        if self.__coordinator is None:
            raise Exception(
                f"method {func.__name__} is decorated with @parallel so it can only be called after calling `register_coordinator`"
            )
        if not self.__coordinator.is_ready_to_broadcast():
            raise Exception(
                f"method {func.__name__} is decorated with @parallel so it can only be called by the leader node inside a `with coordinator` block"
            )

        # The command to execute the method is broadcasted to all nodes
        # only by the leader node and only if it is not already inside
        # a parallel block.
        # This ensures that follower nodes don't rebroadcast the command
        # and that calls to parallel functions inside parallel functions
        # are not executed multiple times.
        if self.__coordinator.is_leader() and not self.__is_inside_parallel_block:
            self.__is_inside_parallel_block = True

            self.__coordinator.broadcast(
                self.__namespace, func.__name__, *args, **kwargs
            )
            result = func(self, *args, **kwargs)

            self.__is_inside_parallel_block = False

            return result
        else:
            return func(self, *args, **kwargs)


# Function decorator that can be used to transform a method
# of a class inheriting from Coordinated into a parallel method.
def parallel(func: Callable) -> Callable:
    def wrapper(self: Coordinated, *args, **kwargs):
        return self.run_method_in_parallel(func, *args, **kwargs)

    return wrapper
