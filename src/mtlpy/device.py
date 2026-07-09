from __future__ import annotations
import numpy as np
from .buffer import Buffer
from .pipeline import Pipeline
from .texture import Sampler, Texture
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
            buf = self.empty(arr.shape, arr.dtype)
            buf.contents[:] = arr.reshape(-1)  # buf.contents is always flat
            return buf
        size = int(data)
        dt   = utils.to_numpy(dtype)
        raw  = self._dev.create_buffer(size * np.dtype(dt).itemsize)
        return Buffer(raw, dt, (size,), self)

    def empty(self, size: int | tuple[int, ...], dtype) -> Buffer:
        shape = (int(size),) if isinstance(size, (int, np.integer)) else tuple(int(s) for s in size)
        flat_size = utils.shape_size(shape)
        dt  = utils.to_numpy(dtype)
        raw = self._dev.create_buffer(flat_size * np.dtype(dt).itemsize)
        return Buffer(raw, dt, shape, self)

    def compile(self, source: str, function_name: str) -> Pipeline:
        return Pipeline(self._dev.compile(source, function_name))

    def empty_texture(self, shape: tuple[int, ...], pixel_format: str) -> Texture:
        """shape is the *spatial* shape only -- (width,), (height, width),
        or (depth, height, width) for a 1D/2D/3D texture; channel count
        comes from pixel_format (e.g. "rgba8Unorm" is 4-channel) and isn't
        part of shape. len(shape) determines dims."""
        dims = len(shape)
        if dims not in (1, 2, 3):
            raise ValueError(
                f"Texture shape must have 1, 2, or 3 dims (spatial only -- "
                f"exclude the channel axis, which pixel_format implies), got {shape}"
            )
        info = utils.pixel_format_info(pixel_format)
        if info.channels > 1 and dims == 3 and shape[-1] == info.channels:
            raise ValueError(
                f"shape {shape} looks like it includes a trailing channel axis "
                f"-- pixel_format {pixel_format!r} already implies {info.channels} "
                f"channels, so shape should be spatial dims only (e.g. shape[:-1]). "
                f"Use device.texture(data, pixel_format) to create+upload from an "
                f"array that still has its channel axis."
            )
        width  = shape[-1]
        height = shape[-2] if dims >= 2 else 1
        depth  = shape[-3] if dims >= 3 else 1
        raw = self._dev.create_texture(dims, info.mtl_value, width, height, depth)
        return Texture(raw, dims, pixel_format, width, height, depth, self)

    def texture(self, data: np.ndarray, pixel_format: str) -> Texture:
        """Create a texture matching data's shape and upload it in one call.
        data's last axis is the channel axis if pixel_format is multi-channel
        (e.g. an (H, W, 4) array for "rgba8Unorm"), matching Texture.shape."""
        info = utils.pixel_format_info(pixel_format)
        spatial_shape = data.shape[:-1] if info.channels > 1 else data.shape
        tex = self.empty_texture(spatial_shape, pixel_format)
        tex.upload(data)
        return tex

    def sampler(self, linear: bool = True, repeat: bool = False) -> Sampler:
        """linear=False uses nearest-neighbor filtering; repeat=True wraps
        out-of-bounds texture coordinates instead of clamping to the edge."""
        raw = self._dev.create_sampler(linear, repeat)
        return Sampler(raw, linear, repeat, self)

    def _binary_op(self, name: str, shader_fn, a: Buffer, b: Buffer, out: Buffer | None = None) -> Buffer:
        if a._device is not b._device:
            raise ValueError(
                "Buffers belong to different Device instances -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        # Checked by .size, not .shape: a Metal buffer has no shape of its
        # own (it's just bytes) -- .shape is Python-side metadata, and two
        # equal-size buffers with different declared shapes are still
        # perfectly valid operands.
        if a.size != b.size:
            raise ValueError(f"Buffer size mismatch: {a.size} != {b.size}")
        if a.dtype != b.dtype:
            raise TypeError(f"Buffer dtype mismatch: {a.dtype} != {b.dtype}")
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        if out is None:
            out = self.empty(a.shape, a.dtype)
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
            out = self.empty(a.shape, a.dtype)
        pipeline.run([a, scalar_buf, out], a.size)
        return out

    def _negate_op(self, a: Buffer) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader.negate_kernel(metal_type), "negate")
        out        = self.empty(a.shape, a.dtype)
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
        out        = self.empty(a.shape, np.bool_)
        pipeline.run([a, b, out], a.size)
        return out

    def _compare_scalar_op(self, name: str, shader_fn, a: Buffer, scalar) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        scalar_buf = self.buffer(np.array([scalar], dtype=a.dtype))
        out        = self.empty(a.shape, np.bool_)
        pipeline.run([a, scalar_buf, out], a.size)
        return out

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._dev.max_threads_per_threadgroup()
