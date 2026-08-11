from __future__ import annotations


class Pipeline:
    def __init__(self, _pipeline):
        self._pipeline = _pipeline  # _mtlpy.Pipeline

    def run(self, buffers: list, grid, wait: bool = True,
            textures: list | None = None, samplers: list | None = None,
            threadgroup=None) -> tuple[float, float]:
        """Returns (gpu_start, gpu_end) in seconds -- pure device-side
        execution time for this dispatch (MTLCommandBuffer's GPUStartTime/
        GPUEndTime), excluding CPU-side encoding/dispatch overhead. Only
        meaningful when wait=True; (0.0, 0.0) when wait=False, since the
        command buffer isn't guaranteed to have even started on the GPU yet.

        threadgroup optionally overrides the threads-per-threadgroup size
        Metal would otherwise pick automatically (an int, or a 1-to-3-element
        sequence -- missing trailing dims are padded with 1, same convention
        as grid). Useful when a kernel's correctness or performance depends
        on a specific threadgroup shape, e.g. one matching a
        threadgroup-memory tile size. Must satisfy both of Metal's own
        constraints or this raises: total threads (w*h*d) <=
        max_threads_per_threadgroup, and a multiple of
        thread_execution_width. Leave as None (the default) to keep the
        existing auto-computed size."""
        if isinstance(grid, int):
            grid = [grid, 1, 1]
        tg = None
        if threadgroup is not None:
            if isinstance(threadgroup, int):
                threadgroup = [threadgroup]
            tg = list(threadgroup) + [1] * (3 - len(threadgroup))
        return self._pipeline.run(
            [b._buf for b in buffers],
            [t._tex for t in (textures or [])],
            [s._sampler for s in (samplers or [])],
            list(grid),
            wait,
            tg,
        )

    @property
    def thread_execution_width(self) -> int:
        return self._pipeline.thread_execution_width()

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._pipeline.max_threads_per_threadgroup()
