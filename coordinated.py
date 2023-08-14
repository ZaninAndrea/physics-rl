from mpi4py import MPI
import random as rand


def random(comm):
    return comm.bcast(rand.random(), root=0)


def randint(comm, a, b):
    return comm.bcast(rand.randint(a, b), root=0)


class Coordinated(object):
    def register_coordinator(self, coordinator):
        self.coordinator_registration = coordinator.register(self)

    def get_coordinator_registration(self):
        return self.coordinator_registration


class CoordinatorRegistration(object):
    def __init__(self, coordinator, namespace):
        super(CoordinatorRegistration, self).__init__()

        self.coordinator = coordinator
        self.namespace = namespace

    def broadcast(self, method, *args, **kwargs):
        self.coordinator.broadcast(self.namespace, method, *args, **kwargs)

    def enter_parallel_block(self):
        self.coordinator.is_inside_parallel_block = True

    def exit_parallel_block(self):
        self.coordinator.is_inside_parallel_block = False

    def is_inside_parallel_block(self):
        return self.coordinator.is_inside_parallel_block

    def is_leader(self):
        return self.coordinator.is_leader()

    def is_leader_inside_with_statement(self):
        return self.coordinator.is_leader_inside_with_statement


class Coordinator(object):
    def __init__(self, comm):
        super(Coordinator, self).__init__()

        self.comm = comm
        self.objects = {}
        self.is_inside_parallel_block = False
        self.is_leader_inside_with_statement = False
        self.registrationCounter = 0

    def register(self, object, namespace=None):
        if namespace is None:
            namespace = "object{}".format(self.registrationCounter)

        self.objects[namespace] = object
        self.registrationCounter += 1

        return CoordinatorRegistration(self, namespace)

    def broadcast(self, namespace, method, *args, **kwargs):
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

    def __enter__(self):
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

    def __exit__(self, *args):
        if self.comm.rank == 0:
            self.comm.bcast("stop", root=0)
            self.is_leader_inside_with_statement = False

    def is_leader(self):
        return self.comm.rank == 0


def parallel(func):
    def wrapper(self, *args, **kwargs):
        coordinator_registration = self.get_coordinator_registration()
        if not coordinator_registration.is_leader_inside_with_statement():
            raise Exception(
                f"method {func.__name__} is decorated with @parallel so it can only be called by the leader node inside a `with coordinator` block"
            )

        if (
            coordinator_registration.is_leader()
            and not coordinator_registration.is_inside_parallel_block()
        ):
            coordinator_registration.enter_parallel_block()

            coordinator_registration.broadcast(func.__name__, *args, **kwargs)
            result = func(self, *args, **kwargs)

            coordinator_registration.exit_parallel_block()

            return result
        else:
            return func(self, *args, **kwargs)

    return wrapper
