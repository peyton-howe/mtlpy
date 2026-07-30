"""Buffer.__dlpack__/__dlpack_device__ (src/mtlpy/buffer.py) export a Buffer
as a DLPack capsule tagged kDLMetal, backed directly by the underlying
id<MTLBuffer>. mtlpy itself never imports MLX (or any other DLPack
consumer) anywhere -- the dunder protocol methods are the complete surface;
a consumer's own entry point calls them automatically. mx.asarray(buf,
copy=...) is what's used below -- unlike mx.array()'s constructor (whose
signature has no copy= parameter at all), mx.asarray's docstring explicitly
documents DLPack-aware copy semantics ("share memory when possible, copy
otherwise"), and it's the one verified stable across repeated calls on the
same Buffer in this session (repeated mx.array(buf) calls in a loop hung --
not root-caused, mx.asarray doesn't exhibit it). Tested two ways: protocol-
level (no MLX needed) and, where MLX happens to be installed, an actual
zero-copy import through it as one concrete consumer.
"""
import gc

import numpy as np
import pytest

try:
    import mtlpy
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_dlpack_device_is_metal(device):
    buf = device.buffer(np.zeros(4, dtype=np.float32))
    assert buf.__dlpack_device__() == (8, 0)  # kDLMetal, device 0


def test_dlpack_returns_a_capsule(device):
    buf = device.buffer(np.zeros(4, dtype=np.float32))
    cap = buf.__dlpack__()
    assert type(cap).__name__ == "PyCapsule"


def test_dlpack_copy_true_raises(device):
    buf = device.buffer(np.zeros(4, dtype=np.float32))
    with pytest.raises(BufferError):
        buf.__dlpack__(copy=True)


def test_dlpack_wrong_device_raises(device):
    buf = device.buffer(np.zeros(4, dtype=np.float32))
    with pytest.raises(BufferError):
        buf.__dlpack__(dl_device=(1, 0))  # kDLCPU -- not what this exports


def test_dlpack_unconsumed_capsule_is_collected_cleanly(device):
    """A capsule that's created but never handed to a DLPack consumer must
    still release the MTL::Buffer retain taken in _dlpack_capsule (see
    bindings.cpp) -- this just exercises that path without crashing/leaking
    an assertion; there's no Python-visible refcount to check directly."""
    buf = device.buffer(np.zeros(4, dtype=np.float32))
    cap = buf.__dlpack__()
    del cap
    gc.collect()


mx = pytest.importorskip("mlx.core", reason="MLX not installed")


def test_mlx_asarray_is_zero_copy(device):
    buf = device.buffer(np.arange(8, dtype=np.float32))
    arr = mx.asarray(buf, copy=False)
    mx.eval(arr)

    buf.contents[0] = 111.0
    buf.contents[7] = 222.0
    mx.eval(arr)

    assert arr[0].item() == 111.0
    assert arr[7].item() == 222.0


def test_mlx_asarray_default_copy_is_still_zero_copy(device):
    """copy=None (mx.asarray's default) is documented as "share memory when
    possible, copy otherwise" -- since a Buffer is always kDLMetal/Shared
    (see __dlpack__), it should still come through zero-copy without
    needing copy=False explicitly."""
    buf = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    arr = mx.asarray(buf)
    mx.eval(arr)
    buf.contents[0] = 42.0
    mx.eval(arr)
    assert arr[0].item() == 42.0


def test_mlx_asarray_preserves_shape(device):
    buf = device.empty((2, 3), np.float32)
    buf.numpy()[:] = np.arange(6, dtype=np.float32).reshape(2, 3)
    arr = mx.asarray(buf, copy=False)
    mx.eval(arr)
    assert tuple(arr.shape) == (2, 3)
    np.testing.assert_array_equal(np.array(arr), buf.numpy())


@pytest.mark.parametrize("dtype", [
    np.float32, np.float16, np.int32, np.uint32, np.int16, np.uint16, np.bool_,
])
def test_mlx_asarray_dtype_coverage(device, dtype):
    buf = device.buffer(np.array([1, 0, 1, 0], dtype=dtype))
    arr = mx.asarray(buf, copy=False)
    mx.eval(arr)
    np.testing.assert_array_equal(np.array(arr).astype(dtype), buf.numpy())
