from __future__ import annotations
import ctypes
import numpy as np
from . import shader, utils


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
        arr        = np.ctypeslib.as_array(ctypes_arr).view(self.dtype)
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

    def __add__(self, other: Buffer) -> Buffer:
        return self._device._binary_op("add", shader.add_kernel, self, other)

    def __sub__(self, other: Buffer) -> Buffer:
        return self._device._binary_op("sub", shader.sub_kernel, self, other)

    def __mul__(self, other: Buffer) -> Buffer:
        return self._device._binary_op("mul", shader.mul_kernel, self, other)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"Buffer(size={self.size}, dtype={self.dtype})"
