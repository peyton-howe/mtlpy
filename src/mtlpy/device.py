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


def list_devices() -> list[str]:
    """Names of all Metal-capable GPUs on this machine, in the order
    Device(index=...) expects. On most Macs (a single integrated GPU) this
    returns exactly one name; multi-GPU Macs (e.g. with an eGPU) list more."""
    return _mtlpy.list_devices()


class Device:
    def __init__(self, index: int | None = None):
        """index selects a specific GPU from list_devices() (for multi-GPU
        machines); the default (None) uses the system default GPU."""
        self._dev = _mtlpy.Device(-1 if index is None else index)

    def __enter__(self) -> Device:
        return self

    def __exit__(self, *exc_info) -> None:
        self.flush_cache()

    def flush_cache(self) -> None:
        """Serialize the on-disk compiled-pipeline cache now, rather than
        waiting for this Device to be garbage collected. Useful for a
        long-running process that wants newly-compiled kernels to survive
        a crash, or just deterministic cleanup via `with mtlpy.Device() as d:`."""
        self._dev.flush_cache()

    def buffer(self, data: np.ndarray | int, dtype=None) -> Buffer:
        if isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data)
            buf = self.empty(arr.size, arr.dtype)
            buf.contents[:] = arr.reshape(-1)  # buf.contents is always flat
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

    def _binary_op(self, name: str, shader_fn, a: Buffer, b: Buffer, out: Buffer | None = None) -> Buffer:
        if a._device is not b._device:
            raise ValueError(
                "Buffers belong to different Device instances -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        if a.size != b.size:
            raise ValueError(f"Buffer size mismatch: {a.size} != {b.size}")
        if a.dtype != b.dtype:
            raise TypeError(f"Buffer dtype mismatch: {a.dtype} != {b.dtype}")
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        if out is None:
            out = self.empty(a.size, a.dtype)
        # Safe to alias out with a/b: each GPU thread reads then writes only
        # its own index, so an in-place dispatch (out is a or b) has no
        # cross-thread data hazard.
        pipeline.run([a, b, out], a.size)
        return out

    def _scalar_op(self, name: str, shader_fn, a: Buffer, scalar, out: Buffer | None = None) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        scalar_buf = self.buffer(np.array([scalar], dtype=a.dtype))
        if out is None:
            out = self.empty(a.size, a.dtype)
        pipeline.run([a, scalar_buf, out], a.size)
        return out

    def _negate_op(self, a: Buffer) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader.negate_kernel(metal_type), "negate")
        out        = self.empty(a.size, a.dtype)
        pipeline.run([a, out], a.size)
        return out

    def _compare_op(self, name: str, shader_fn, a: Buffer, b: Buffer) -> Buffer:
        if a._device is not b._device:
            raise ValueError(
                "Buffers belong to different Device instances -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        if a.size != b.size:
            raise ValueError(f"Buffer size mismatch: {a.size} != {b.size}")
        if a.dtype != b.dtype:
            raise TypeError(f"Buffer dtype mismatch: {a.dtype} != {b.dtype}")
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        out        = self.empty(a.size, np.bool_)
        pipeline.run([a, b, out], a.size)
        return out

    def _compare_scalar_op(self, name: str, shader_fn, a: Buffer, scalar) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        scalar_buf = self.buffer(np.array([scalar], dtype=a.dtype))
        out        = self.empty(a.size, np.bool_)
        pipeline.run([a, scalar_buf, out], a.size)
        return out

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._dev.max_threads_per_threadgroup()
