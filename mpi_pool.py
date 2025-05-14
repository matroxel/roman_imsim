class _close_pool_message(object):
    def __repr__(self):
        return "<Close pool message>"


class _function_wrapper(object):
    def __init__(self, function, callback=None):
        self.function = function
        self.callback = callback


def _error_function(task):
    raise RuntimeError("Pool was sent tasks before being told what " "function to apply.")


class MPIPool(object):
    def __init__(self, debug=False, comm=None):
        try:
            from mpi4py import MPI

            self.MPI = MPI
        except ImportError:
            raise RuntimeError("MPI environment not found!")
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.debug = debug

        self.function = _error_function
        self.callback = None

    def is_master(self):
        return self.rank == 0

    def wait(self):
        if self.is_master():
            raise RuntimeError("Master node told to await jobs")
        status = self.MPI.Status()
        while True:
            task = self.comm.recv(source=0, tag=self.MPI.ANY_TAG, status=status)

            if isinstance(task, _close_pool_message):
                break

            if isinstance(task, _function_wrapper):
                self.function = task.function
                self.callback = task.callback
                continue

            if self.callback:

                def compose(x):
                    result = self.function(x)
                    self.callback(x, result)
                    return result

                results = map(compose, task)
            else:
                results = map(self.function, task)
            self.comm.send(results, dest=0, tag=status.tag)

    def map(self, function, tasks, callback=None):
        # Should be called by the master only
        if not self.is_master():
            self.wait()
            return

        # send function if necessary
        if function is not self.function or callback is not self.callback:
            self.function = function
            self.callback = callback
            F = _function_wrapper(function, callback)
            requests = [self.comm.send(F, dest=i) for i in range(1, self.size)]
            # self.MPI.Request.waitall(requests)

        # distribute tasks to workers
        requests = []
        for i in range(1, self.size):
            req = self.comm.send(tasks[i :: self.size], dest=i)
            requests.append(req)

        # process local work
        results = [None] * len(tasks)

        if self.callback:

            def compose(x):
                result = self.function(x)
                self.callback(x, result)
                return result

            results[:: self.size] = map(compose, tasks[:: self.size])
        else:
            results[:: self.size] = map(self.function, tasks[:: self.size])

        # recover results from workers (in any order)
        status = self.MPI.Status()
        for i in range(self.size - 1):
            result = self.comm.recv(source=self.MPI.ANY_SOURCE, status=status)
            results[status.source :: self.size] = result
        return results

    def gather(self, data, root=0):
        return self.comm.gather(data, root)

    def bcast(self, data, root=0):
        return self.comm.bcast(data, root)

    def send(self, data, dest=0, tag=0):
        self.comm.send(data, dest, tag)

    def recv(self, source=0, tag=0):
        return self.comm.recv(source, tag)

    def close(self):
        if self.is_master():
            for i in range(1, self.size):
                self.comm.isend(_close_pool_message(), dest=i)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
