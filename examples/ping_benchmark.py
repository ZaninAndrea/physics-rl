# This is a dummy environment that only supports a ping operation.
# This environment is used to measure the latency of executing a single
# operation in a coordinated environment.

from physicsrl import coordinated
import time


class PingEnvironment(coordinated.Coordinated):
    def __init__(self, coordinator: coordinated.Coordinator) -> None:
        self.register_coordinator(coordinator)

    @coordinated.parallel
    def ping(self) -> None:
        pass


coordinator = coordinated.Coordinator()
env = PingEnvironment(coordinator)

with coordinator:
    if coordinator.is_leader():
        start_time = time.time()

        for i in range(10_000_000):
            env.ping()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        # Save execution time in a file
        with open("python_times.txt", "a") as file:
            file.write(str(execution_time) + "\n")
