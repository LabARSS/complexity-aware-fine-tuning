import multiprocessing as mp


def reset_memory_after_completion(target, *args, **kwargs):
    ctx = mp.get_context("spawn")  # safe with CUDA
    q = ctx.Queue()
    p = ctx.Process(target=target, args=args, kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"stage crashed with {p.exitcode}")
    return q.get()

