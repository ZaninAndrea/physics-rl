from mpi4py import MPI


class CoordinatedContext:
    def __init__(self, comm, methods, object):
        self.comm = comm
        self.methods = methods
        self.object = object

    def __enter__(self):
        if self.comm.rank != 0:
            while True:
                command = self.comm.bcast(None, root=0)
                if command == "stop":
                    break
                else:
                    method = getattr(self.object, command["method"])
                    method(*command["args"], **command["kwargs"])

            return None
        else:
            wrapped_object = self.object
            methods = self.methods
            comm = self.comm

            class DecoratedClass:
                def __getattribute__(self, name):
                    attr = getattr(wrapped_object, name)

                    if hasattr(attr, "__call__") and name in methods:
                        # Replace the implementation of the coordinated methods
                        # with a new function that broadcasts the call

                        def newfunc(*args, **kwargs):
                            comm.bcast(
                                {"method": name, "args": args, "kwargs": kwargs}, root=0
                            )
                            return attr(*args, **kwargs)

                        return newfunc
                    else:
                        return attr

            return DecoratedClass()

    def __exit__(self, *args):
        if self.comm.rank == 0:
            self.comm.bcast("stop", root=0)


class CoordinatedPyEnvironment(CoordinatedContext):
    def __init__(self, comm, env):
        super().__init__(comm, ["step", "reset"], env)


# coordinated returns a decorator that wraps any class
# and broadcasts any method call made on the rank 0
# instance to all the other processes
def coordinated(comm, methods):
    if "_wrapped_object" in methods:
        raise Exception(
            "Cannot use _wrapped_object as a coordinated method, it is reserved"
        )

    rank = comm.Get_rank()

    def decorator(wrapped_class):
        class DecoratedClass(object):
            def __init__(self, *args, **kwargs):
                self._wrapped_object = wrapped_class(*args, **kwargs)

                if comm.rank != 0:
                    while True:
                        command = comm.bcast(None, root=0)
                        if command == "stop":
                            break
                        else:
                            method = getattr(self._wrapped_object, command["method"])
                            method(*command["args"], **command["kwargs"])

            def __getattribute__(self, name):
                if name in ["_wrapped_object"]:
                    return object.__getattribute__(self, name)

                attr = getattr(self._wrapped_object, name)

                # Replace the implementation of the coordinated methods
                # with a new function that broadcasts the call
                if hasattr(attr, "__call__") and name in methods:

                    def newfunc(*args, **kwargs):
                        if rank != 0:
                            raise Exception(
                                "Cannot call coordinated methods on processes with rank not equal to 0"
                            )

                        comm.bcast(
                            {"method": name, "args": args, "kwargs": kwargs}, root=0
                        )
                        return attr(*args, **kwargs)

                    return newfunc
                else:
                    return attr

        return DecoratedClass

    return decorator


def stop_coordination(comm):
    comm.bcast("stop", root=0)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    @coordinated(comm, ["ping"])
    class Worker:
        def ping(self, text):
            self.pong(text)

        def pong(self, text):
            print(text)

    w = Worker()

    if rank == 0:
        w.ping("prova")
        w.stop_coordination()
