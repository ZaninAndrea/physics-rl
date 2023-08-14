from mpi4py import MPI
from tf_agents.environments import py_environment
import random as rand


def random(comm):
    return comm.bcast(rand.random(), root=0)


def randint(comm, a, b):
    return comm.bcast(rand.randint(a, b), root=0)


class Coordinator:
    def __init__(self, comm):
        self.comm = comm
        self.objects = {}

    def register(self, object, namespace):
        self.objects[namespace] = object

        def broadcast(method, *args, **kwargs):
            if self.comm.rank == 0:
                self.comm.bcast(
                    {
                        "namespace": namespace,
                        "method": method,
                        "args": args,
                        "kwargs": kwargs,
                    },
                    root=0,
                )

        return broadcast

    def __enter__(self):
        if self.comm.rank != 0:
            while True:
                command = self.comm.bcast(None, root=0)
                if command == "stop":
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

    def __exit__(self, *args):
        if self.comm.rank == 0:
            self.comm.bcast("stop", root=0)

    def is_leader(self):
        return self.comm.rank == 0


class CoordinatedPyEnvironment(py_environment.PyEnvironment):
    def __init__(self, comm):
        super(CoordinatedPyEnvironment, self).__init__()
        self.comm = comm

    def __enter__(self):
        if self.comm.rank != 0:
            while True:
                command = self.comm.bcast(None, root=0)
                if command == "stop":
                    break
                else:
                    method = getattr(self, command["method"])
                    method(*command["args"], **command["kwargs"])

            return None
        else:
            parentObject = self
            comm = self.comm

            class DecoratedClass(py_environment.PyEnvironment):
                def action_spec(self):
                    return parentObject.action_spec()

                def observation_spec(self):
                    return parentObject.observation_spec()

                def _reset(self, *args, **kwargs):
                    comm.bcast(
                        {"method": "_reset", "args": args, "kwargs": kwargs}, root=0
                    )
                    return parentObject._reset(*args, **kwargs)

                def _step(self, *args, **kwargs):
                    comm.bcast(
                        {"method": "_step", "args": args, "kwargs": kwargs}, root=0
                    )
                    return parentObject._step(*args, **kwargs)

            return DecoratedClass()

    def __exit__(self, *args):
        if self.comm.rank == 0:
            self.comm.bcast("stop", root=0)
