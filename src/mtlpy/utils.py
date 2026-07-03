import numpy as np

# (metal_type_string, numpy_dtype)
_TABLE = [
    ("float",  np.float32),
    ("double", np.float64),
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
    return np.dtype(hint)
