from __future__ import annotations


class Pipeline:
    def __init__(self, _pipeline):
        self._pipeline = _pipeline  # _mtlpy.Pipeline

    def run(self, buffers: list, grid, wait: bool = True,
            textures: list | None = None, samplers: list | None = None) -> None:
        if isinstance(grid, int):
            grid = [grid, 1, 1]
        self._pipeline.run(
            [b._buf for b in buffers],
            [t._tex for t in (textures or [])],
            [s._sampler for s in (samplers or [])],
            list(grid),
            wait,
        )

    @property
    def thread_execution_width(self) -> int:
        return self._pipeline.thread_execution_width()

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._pipeline.max_threads_per_threadgroup()
