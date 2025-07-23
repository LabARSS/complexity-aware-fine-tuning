import multiprocessing as mp


def reset_memory_after_completion(target):
    def _wrapped_in_worker(**kwargs):
        def _stage_worker(q):
            ckpt = target(**kwargs)
            q.put(ckpt)

        ctx = mp.get_context("spawn")  # safe with CUDA
        q = ctx.Queue()
        p = ctx.Process(target=_stage_worker, args=(q, kwargs))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"stage crashed with {p.exitcode}")
        return q.get()

    return _wrapped_in_worker
