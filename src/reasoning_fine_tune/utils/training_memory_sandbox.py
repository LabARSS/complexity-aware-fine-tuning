import multiprocessing as mp
from functools import wraps


def memory_sandbox_worker(target):
    @wraps(target)
    def wrapped(q, *args, **kwargs):
        res = target(*args, **kwargs)
        q.put(res)

    return wrapped


def run_in_memory_sandbox(target, *args, **kwargs):
    ctx = mp.get_context("spawn")  # safe with CUDA
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q, *args), kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"stage crashed with {p.exitcode}")
    return q.get()
