from __future__ import annotations
import ctypes
import numpy as np
from . import shader, utils


class _BackedArray(np.ndarray):
    """ndarray subclass with a __dict__, so it can hold a _mtlpy_buf backref."""


class Buffer:
    def __init__(self, _buf, dtype: np.dtype, size: int, device):
        self._buf    = _buf             # _mtlpy.Buffer
        self.dtype   = np.dtype(dtype)
        self.size    = size             # element count
        self._device = device           # Python Device (needed for ops)

    @property
    def contents(self) -> np.ndarray:
        nbytes     = self.size * self.dtype.itemsize
        ctypes_arr = (ctypes.c_byte * nbytes).from_address(self._buf.data_ptr)
        arr        = np.ctypeslib.as_array(ctypes_arr).view(self.dtype).view(_BackedArray)
        arr._mtlpy_buf = self           # keep Buffer alive while array is alive
        return arr

    def astype(self, dtype) -> Buffer:
        dst_dtype  = utils.to_numpy(dtype)
        src_metal  = utils.to_metal(self.dtype)
        dst_metal  = utils.to_metal(dst_dtype)
        source     = shader.cast_kernel(src_metal, dst_metal)
        pipeline   = self._device.compile(source, "cast")
        out        = self._device.empty(self.size, dst_dtype)
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
        return f"Buffer(size={self.size}, dtype={self.dtype})"
