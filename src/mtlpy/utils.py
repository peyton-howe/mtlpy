from typing import NamedTuple
import numpy as np


def shape_size(shape: tuple[int, ...]) -> int:
    """Flat element count for a shape tuple (empty tuple, i.e. a 0-d shape,
    is 1 element, matching NumPy's convention for scalars)."""
    return int(np.prod(shape)) if shape else 1

# (metal_type_string, numpy_dtype)
_TABLE = [
    ("float",  np.float32),
    ("half",   np.float16),
    ("int",    np.int32),
    ("uint",   np.uint32),
    ("short",  np.int16),
    ("ushort", np.uint16),
    ("long",   np.int64),
    ("ulong",  np.uint64),
    ("bool",   np.bool_),
]

_NUMPY_TO_METAL = {np.dtype(np_t): metal for metal, np_t in _TABLE}
_METAL_TO_NUMPY = {metal: np.dtype(np_t) for metal, np_t in _TABLE}


def to_metal(dtype) -> str:
    dt = np.dtype(dtype)
    try:
        return _NUMPY_TO_METAL[dt]
    except KeyError:
        raise TypeError(f"No Metal equivalent for dtype: {dt}")


def to_numpy(hint) -> np.dtype:
    if isinstance(hint, str):
        try:
            return _METAL_TO_NUMPY[hint]
        except KeyError:
            raise TypeError(f"Unknown Metal type string: '{hint}'")
    dt = np.dtype(hint)
    if dt == np.float64:
        # Metal Shading Language has no double-precision type on any Apple
        # GPU, so there's no kernel we could ever compile for float64 --
        # downcast at the boundary instead of failing at shader-compile time.
        return np.dtype(np.float32)
    return dt


# (name, MTL::PixelFormat raw value, channels, numpy dtype per channel,
#  normalized (Unorm formats read/write as float in [0,1] despite integer
#  storage), MSL scalar type used to read/write a texel in a kernel).
# Values are from metal-cpp's Metal/MTLPixelFormat.hpp -- a small,
# deliberately non-exhaustive set covering 8-bit and float image data
# (Metal defines >100 pixel formats; most are for graphics, not compute).
_PIXEL_FORMAT_TABLE = [
    ("r8Unorm",     10,  1, np.uint8,   True,  "float"),
    ("rgba8Unorm",  70,  4, np.uint8,   True,  "float"),
    ("r8Uint",      13,  1, np.uint8,   False, "uint"),
    ("rgba8Uint",   73,  4, np.uint8,   False, "uint"),
    ("r16Float",    25,  1, np.float16, False, "half"),
    ("rgba16Float", 115, 4, np.float16, False, "half"),
    ("r32Float",    55,  1, np.float32, False, "float"),
    ("rgba32Float", 125, 4, np.float32, False, "float"),
    ("r32Uint",     53,  1, np.uint32,  False, "uint"),
    ("rgba32Uint",  123, 4, np.uint32,  False, "uint"),
]
class PixelFormatInfo(NamedTuple):
    mtl_value: int
    channels: int
    dtype: np.dtype
    normalized: bool
    msl_scalar_type: str


_PIXEL_FORMATS = {
    name: PixelFormatInfo(value, channels, np.dtype(dt), normalized, msl_t)
    for name, value, channels, dt, normalized, msl_t in _PIXEL_FORMAT_TABLE
}


def pixel_format_info(name: str) -> PixelFormatInfo:
    try:
        return _PIXEL_FORMATS[name]
    except KeyError:
        raise ValueError(
            f"Unknown pixel format: '{name}'. Supported: {sorted(_PIXEL_FORMATS)}"
        )


# MSL type name matching a dtype's storage width exactly -- unlike to_metal()
# above (which maps to a *register*-width type for elementwise Buffer ops,
# and has no uint8 entry at all, since there's no 8-bit MSL arithmetic type),
# this is for packing a buffer at a pixel format's actual per-texel storage
# width, which is narrower than the type MSL widens texture reads to (e.g.
# an 8-bit Uint texture reads as "uint" in a kernel, but stores as "uchar").
_STORAGE_TYPE = {
    np.dtype(np.uint8):   "uchar",
    np.dtype(np.int8):    "char",
    np.dtype(np.uint16):  "ushort",
    np.dtype(np.int16):   "short",
    np.dtype(np.uint32):  "uint",
    np.dtype(np.int32):   "int",
    np.dtype(np.float16): "half",
    np.dtype(np.float32): "float",
}


def msl_storage_type(dtype) -> str:
    dt = np.dtype(dtype)
    try:
        return _STORAGE_TYPE[dt]
    except KeyError:
        raise TypeError(f"No MSL storage type for dtype: {dt}")
