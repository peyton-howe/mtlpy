from __future__ import annotations
import ctypes
import numpy as np
from . import shader, utils
from .utils import StorageMode


class _BackedArray(np.ndarray):
    """ndarray subclass with a __dict__, so it can hold a _mtlpy_buf backref."""


_DLPACK_DEVICE = (8, 0)  # (DLDeviceType.kDLMetal, device 0) -- see csrc/dlpack.h


class Buffer:
    def __init__(self, _buf, dtype: np.dtype, shape: tuple[int, ...], device):
        self._buf    = _buf             # _mtlpy.Buffer
        self.dtype   = np.dtype(dtype)
        self.shape   = tuple(shape)
        self._device = device           # Python Device (needed for ops)
        self.storage = StorageMode(_buf.storage_mode)
        # Computed once here, not as a property recomputed on every access:
        # .shape is never mutated after construction (reshape() below always
        # returns a new Buffer instead), so there's no desync risk from
        # caching it
        self.size    = utils.shape_size(self.shape)  # element count

    def to_storage(self, storage: StorageMode) -> Buffer:
        """A new Buffer holding a copy of this Buffer's data in a different
        storage mode, via a GPU-side blit copy (MTLBlitCommandEncoder,
        wrapped by the internal Device::copy_buffer() -- there's no public
        Device.copy_buffer() Python method, this is the only way to reach
        it) -- unlike a CPU memcpy, this works no matter which storage
        mode(s) are involved (including Private, which has no CPU-visible
        memory at all). Returns self unchanged (no copy) if already in
        storage."""
        storage = StorageMode(storage)
        if storage == self.storage:
            return self
        nbytes = self.size * self.dtype.itemsize
        raw    = self._device._dev.create_buffer(nbytes, int(storage))
        self._device._dev.copy_buffer(self._buf, 0, raw, 0, nbytes, True)
        return Buffer(raw, self.dtype, self.shape, self._device)

    @property
    def contents(self) -> np.ndarray:
        """CPU-visible flat view over this Buffer's data. For a Shared
        Buffer, this is a live, writable, zero-copy view over the actual GPU
        memory -- writes here are writes to what the GPU sees, no separate
        flush needed. A Private/Managed Buffer has no CPU memory that's safe
        to read directly (see StorageMode's docstring), so this
        transparently materializes a Shared copy first via to_storage() --
        safe to call either way, but for those two storage modes the result
        is a disconnected, read-only snapshot: it's marked non-writeable
        (assigning into it raises ValueError) precisely because a write
        there would silently vanish instead of reaching this Buffer's actual
        memory -- neither further GPU writes to this Buffer nor writes to
        the returned array could ever affect the other. To write into a
        Private/Managed Buffer, upload into a Shared staging Buffer and
        .to_storage() it, or write via a compute kernel.

        Calling this repeatedly on the same Private/Managed Buffer redoes
        the full allocate + blit-copy + wait round trip every time (nothing
        is cached on this Buffer) -- if you need the data more than once,
        save the returned array instead of calling .contents/.numpy() again."""
        buf        = self.to_storage(StorageMode.SHARED)
        nbytes     = buf.size * buf.dtype.itemsize
        ctypes_arr = (ctypes.c_byte * nbytes).from_address(buf._buf.data_ptr)
        arr        = np.ctypeslib.as_array(ctypes_arr).view(buf.dtype).view(_BackedArray)
        arr._mtlpy_buf = buf            # keep the (possibly materialized) Buffer alive
        if buf is not self:
            # buf is a throwaway snapshot to_storage() just materialized --
            # writing into it would silently never reach self's real memory,
            # so make that impossible instead of a silent no-op.
            arr.flags.writeable = False
        return arr

    def numpy(self) -> np.ndarray:
        """Contents reshaped to this Buffer's .shape -- unlike .contents
        (always flat, see the property above), this looks like the array
        you created the Buffer from. Zero-copy and writable for a Shared
        Buffer (reshaping a flat contiguous array is always a view, never a
        copy, so this is just as live as .contents); for Private/Managed
        this carries .contents' read-only-snapshot caveat instead."""
        return self.contents.reshape(self.shape)

    @property
    def mtl_ptr(self) -> int:
        """The id<MTLBuffer> handle itself, as a raw integer -- for handing
        this Buffer to code outside mtlpy that wants to do its own native
        Metal interop (see Texture.mtl_ptr's docstring for the general
        pattern and lifetime caveat this shares). For any DLPack-aware
        consumer (MLX, PyTorch, ...), just hand it this Buffer directly --
        e.g. mx.asarray(buf, copy=False) calls __dlpack__ automatically, and
        that path comes with automatic cross-library lifetime management
        (see _dlpack_capsule in bindings.cpp); this raw pointer does not."""
        return self._buf.mtl_ptr

    def __dlpack_device__(self) -> tuple[int, int]:
        return _DLPACK_DEVICE

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """Zero-copy DLPack export, backed directly by this Buffer's
        underlying id<MTLBuffer> (tagged kDLMetal) -- verified against MLX:
        mx.asarray(buf, copy=False) shares this Buffer's live memory rather
        than copying it. Only works for a Shared-storage Buffer (raises
        otherwise -- see StorageMode's docstring): Private storage has no
        CPU-visible memory to hand a DLPack consumer zero-copy, and Managed
        needs an explicit synchronize a generic DLPack consumer won't do.
        Call .to_storage(StorageMode.SHARED) first to get an exportable copy.

        stream is accepted but unused: every op that writes into a Buffer
        (Pipeline.run, elementwise ops, ...) defaults to wait=True and is
        already synchronized by the time it returns. A caller who used
        wait=False or CommandBuffer batching to leave GPU work in flight is
        responsible for waiting before handing the Buffer to another
        framework via DLPack, same as for any other read of .contents.
        """
        if self.storage != StorageMode.SHARED:
            raise BufferError(
                f"Buffer.__dlpack__ requires Shared storage for zero-copy export "
                f"(this Buffer is {self.storage.name}) -- call "
                f".to_storage(mtlpy.StorageMode.SHARED) first to get an exportable copy."
            )
        if copy:
            raise BufferError("Buffer.__dlpack__ does not support copy=True (already zero-copy)")
        if dl_device is not None and dl_device != _DLPACK_DEVICE:
            raise BufferError(f"Cannot export mtlpy Buffer to DLPack device {dl_device}")
        code, bits = utils.to_dlpack_dtype(self.dtype)
        return self._buf._dlpack_capsule(code, bits, list(self.shape))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Lets np.array(buf) / np.asarray(buf) work directly on a Buffer."""
        arr = self.numpy()
        needs_cast = dtype is not None and np.dtype(dtype) != arr.dtype
        if needs_cast and copy is False:
            raise ValueError(
                f"Cannot convert Buffer's dtype ({arr.dtype}) to {np.dtype(dtype)} "
                "without copying, but copy=False was requested"
            )
        if needs_cast:
            return arr.astype(dtype)  # astype() always copies -- satisfies copy=True too
        return arr.copy() if copy else arr

    def reshape(self, *shape) -> Buffer:
        """A new Buffer over the same underlying Metal allocation (no copy,
        no reallocation) with a different logical .shape. Note that
        .contents on the result is still flat -- only .shape/.numpy() see
        the new shape, same as on the original Buffer."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(s) for s in shape)
        new_size = utils.shape_size(shape)
        if new_size != self.size:
            raise ValueError(f"cannot reshape Buffer of size {self.size} into shape {shape}")
        return Buffer(self._buf, self.dtype, shape, self._device)

    def astype(self, dtype) -> Buffer:
        dst_dtype  = utils.to_numpy(dtype)
        src_metal  = utils.to_metal(self.dtype)
        dst_metal  = utils.to_metal(dst_dtype)
        source     = shader.cast_kernel(src_metal, dst_metal)
        pipeline   = self._device.compile(source, "cast")
        out        = self._device.empty(self.shape, dst_dtype, storage=self.storage)
        pipeline.run([self, out], self.size)
        return out

    def __add__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._binary_op("add", shader.add_kernel, self, other)
        return self._device._scalar_op("add_scalar", shader.add_scalar_kernel, self, other)

    def __radd__(self, other) -> Buffer:
        return self.__add__(other)  # addition is commutative

    def __sub__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._binary_op("sub", shader.sub_kernel, self, other)
        return self._device._scalar_op("sub_scalar", shader.sub_scalar_kernel, self, other)

    def __rsub__(self, other) -> Buffer:
        # other - self; not commutative, so this needs its own kernel
        return self._device._scalar_op("rsub_scalar", shader.rsub_scalar_kernel, self, other)

    def __mul__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._binary_op("mul", shader.mul_kernel, self, other)
        return self._device._scalar_op("mul_scalar", shader.mul_scalar_kernel, self, other)

    def __rmul__(self, other) -> Buffer:
        return self.__mul__(other)  # multiplication is commutative

    def __truediv__(self, other) -> Buffer:
        """Elementwise `/`, using Metal's native `/` for self.dtype -- true
        (float) division for float dtypes, but C-style truncating division
        for integer dtypes (unlike NumPy's `/`, which always promotes
        integers to float64). Matches this library's existing philosophy of
        staying a thin, predictable wrapper around Metal rather than
        replicating NumPy's type-promotion rules."""
        if isinstance(other, Buffer):
            return self._device._binary_op("div", shader.div_kernel, self, other)
        return self._device._scalar_op("div_scalar", shader.div_scalar_kernel, self, other)

    def __rtruediv__(self, other) -> Buffer:
        # other / self; not commutative, so this needs its own kernel
        return self._device._scalar_op("rdiv_scalar", shader.rdiv_scalar_kernel, self, other)

    def __neg__(self) -> Buffer:
        return self._device._negate_op(self)

    def __iadd__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._binary_op("add", shader.add_kernel, self, other, out=self)
        return self._device._scalar_op("add_scalar", shader.add_scalar_kernel, self, other, out=self)

    def __isub__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._binary_op("sub", shader.sub_kernel, self, other, out=self)
        return self._device._scalar_op("sub_scalar", shader.sub_scalar_kernel, self, other, out=self)

    def __imul__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._binary_op("mul", shader.mul_kernel, self, other, out=self)
        return self._device._scalar_op("mul_scalar", shader.mul_scalar_kernel, self, other, out=self)

    def __itruediv__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._binary_op("div", shader.div_kernel, self, other, out=self)
        return self._device._scalar_op("div_scalar", shader.div_scalar_kernel, self, other, out=self)

    # Comparisons return a bool Buffer, matching NumPy's ndarray convention
    # rather than Python's identity-comparison convention -- like ndarray,
    # this makes Buffer unhashable (Python clears __hash__ when __eq__ is
    # defined), which is the right tradeoff for an array-like type.
    def __eq__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._compare_op("eq", shader.eq_kernel, self, other)
        return self._device._compare_scalar_op("eq_scalar", shader.eq_scalar_kernel, self, other)

    def __ne__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._compare_op("ne", shader.ne_kernel, self, other)
        return self._device._compare_scalar_op("ne_scalar", shader.ne_scalar_kernel, self, other)

    def __lt__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._compare_op("lt", shader.lt_kernel, self, other)
        return self._device._compare_scalar_op("lt_scalar", shader.lt_scalar_kernel, self, other)

    def __le__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._compare_op("le", shader.le_kernel, self, other)
        return self._device._compare_scalar_op("le_scalar", shader.le_scalar_kernel, self, other)

    def __gt__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._compare_op("gt", shader.gt_kernel, self, other)
        return self._device._compare_scalar_op("gt_scalar", shader.gt_scalar_kernel, self, other)

    def __ge__(self, other) -> Buffer:
        if isinstance(other, Buffer):
            return self._device._compare_op("ge", shader.ge_kernel, self, other)
        return self._device._compare_scalar_op("ge_scalar", shader.ge_scalar_kernel, self, other)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        storage = "" if self.storage == StorageMode.SHARED else f", storage={self.storage.name.lower()}"
        return f"Buffer(shape={self.shape}, dtype={self.dtype}{storage})"
