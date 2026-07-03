from __future__ import annotations
import numpy as np
from .buffer import Buffer
from .pipeline import Pipeline
from . import utils, shader

try:
    from . import _mtlpy
except ImportError as e:
    raise ImportError(
        "mtlpy C extension not found. Install with: pip install mtlpy"
    ) from e


class Device:
    def __init__(self):
        self._dev = _mtlpy.Device()

    def buffer(self, data: np.ndarray | int, dtype=None) -> Buffer:
        if isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data)
            buf = self.empty(arr.size, arr.dtype)
            buf.contents[:] = arr
            return buf
        size = int(data)
        dt   = utils.to_numpy(dtype)
        raw  = self._dev.create_buffer(size * np.dtype(dt).itemsize)
        return Buffer(raw, dt, size, self)

    def empty(self, size: int, dtype) -> Buffer:
        dt  = utils.to_numpy(dtype)
        raw = self._dev.create_buffer(size * np.dtype(dt).itemsize)
        return Buffer(raw, dt, size, self)

    def compile(self, source: str, function_name: str) -> Pipeline:
        return Pipeline(self._dev.compile(source, function_name))

    def _binary_op(self, name: str, shader_fn, a: Buffer, b: Buffer) -> Buffer:
        if a.size != b.size:
            raise ValueError(f"Buffer size mismatch: {a.size} != {b.size}")
        if a.dtype != b.dtype:
            raise TypeError(f"Buffer dtype mismatch: {a.dtype} != {b.dtype}")
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        out        = self.empty(a.size, a.dtype)
        pipeline.run([a, b, out], a.size)
        return out

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._dev.max_threads_per_threadgroup()
