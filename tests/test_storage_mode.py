"""Buffer storage modes (see src/mtlpy/utils.py's StorageMode): SHARED is
the default and behaves exactly as before this feature existed; MANAGED and
PRIVATE have no CPU-visible memory that's safe to read/write directly, so
Buffer.contents/.numpy()/.to_storage() materialize a Shared copy instead.
These tests pin down that materialization, the read-only-snapshot guard on
writes, and that storage mode propagates through ops instead of silently
downgrading to Shared.
"""
import numpy as np
import pytest

try:
    import mtlpy
    from mtlpy import StorageMode
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")

NON_SHARED = [StorageMode.PRIVATE, StorageMode.MANAGED]


def test_default_storage_is_shared(device):
    buf = device.buffer(np.arange(4, dtype=np.float32))
    assert buf.storage == StorageMode.SHARED
    assert device.empty(4, np.float32).storage == StorageMode.SHARED


@pytest.mark.parametrize("storage", NON_SHARED)
def test_buffer_from_ndarray_respects_storage(device, storage):
    data = np.arange(8, dtype=np.float32)
    buf = device.buffer(data, storage=storage)
    assert buf.storage == storage
    np.testing.assert_array_equal(buf.numpy(), data)


@pytest.mark.parametrize("storage", NON_SHARED)
def test_empty_respects_storage(device, storage):
    buf = device.empty(8, np.float32, storage=storage)
    assert buf.storage == storage


@pytest.mark.parametrize("storage", NON_SHARED)
def test_contents_and_numpy_materialize_shared_copy(device, storage):
    data = np.arange(8, dtype=np.float32)
    buf = device.buffer(data, storage=storage)
    np.testing.assert_array_equal(buf.contents, data)
    np.testing.assert_array_equal(buf.numpy(), data)


@pytest.mark.parametrize("storage", NON_SHARED)
def test_contents_write_on_non_shared_raises_instead_of_silently_vanishing(device, storage):
    """A write through .contents on a Private/Managed Buffer used to be a
    silent no-op (it landed in a throwaway materialized copy, never in the
    original Buffer's actual memory) -- it must now raise instead."""
    buf = device.empty(4, np.float32, storage=storage)
    with pytest.raises(ValueError):
        buf.contents[:] = [1.0, 2.0, 3.0, 4.0]


def test_shared_contents_still_writable(device):
    buf = device.empty(4, np.float32)
    buf.contents[:] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(buf.numpy(), [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("storage", list(StorageMode))
def test_to_storage_round_trip(device, storage):
    data = np.arange(6, dtype=np.int32)
    buf = device.buffer(data)  # SHARED
    converted = buf.to_storage(storage)
    assert converted.storage == storage
    back = converted.to_storage(StorageMode.SHARED)
    np.testing.assert_array_equal(back.numpy(), data)


def test_to_storage_is_a_noop_when_already_that_storage(device):
    buf = device.buffer(np.arange(4, dtype=np.float32))
    assert buf.to_storage(StorageMode.SHARED) is buf


@pytest.mark.parametrize("storage", NON_SHARED)
def test_dlpack_rejects_non_shared_buffer(device, storage):
    buf = device.empty(4, np.float32, storage=storage)
    with pytest.raises(BufferError):
        buf.__dlpack__()


def test_dlpack_works_after_to_storage(device):
    buf = device.empty(4, np.float32, storage=StorageMode.PRIVATE).to_storage(StorageMode.SHARED)
    buf.__dlpack__()  # must not raise


@pytest.mark.parametrize("storage", NON_SHARED)
def test_reshape_preserves_storage(device, storage):
    buf = device.empty(8, np.float32, storage=storage)
    assert buf.reshape(2, 4).storage == storage


@pytest.mark.parametrize("storage", NON_SHARED)
def test_astype_preserves_storage(device, storage):
    buf = device.buffer(np.arange(4, dtype=np.float32), storage=storage)
    out = buf.astype(np.int32)
    assert out.storage == storage
    np.testing.assert_array_equal(out.numpy(), [0, 1, 2, 3])


@pytest.mark.parametrize("storage", NON_SHARED)
def test_binary_op_preserves_storage(device, storage):
    a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32), storage=storage)
    b = device.buffer(np.array([4.0, 5.0, 6.0], dtype=np.float32), storage=storage)
    out = a + b
    assert out.storage == storage
    np.testing.assert_array_equal(out.numpy(), [5.0, 7.0, 9.0])


@pytest.mark.parametrize("storage", NON_SHARED)
def test_scalar_op_preserves_storage(device, storage):
    a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32), storage=storage)
    out = a * 2.0
    assert out.storage == storage
    np.testing.assert_array_equal(out.numpy(), [2.0, 4.0, 6.0])


@pytest.mark.parametrize("storage", NON_SHARED)
def test_compare_op_preserves_storage(device, storage):
    a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32), storage=storage)
    b = device.buffer(np.array([1.0, 0.0, 3.0], dtype=np.float32), storage=storage)
    out = a == b
    assert out.storage == storage
    np.testing.assert_array_equal(out.numpy(), [True, False, True])


def test_repr_shows_non_shared_storage(device):
    shared = device.empty(4, np.float32)
    private = device.empty(4, np.float32, storage=StorageMode.PRIVATE)
    assert "storage=" not in repr(shared)
    assert "storage=private" in repr(private)


def test_copy_buffer_out_of_range_raises_instead_of_crashing(device):
    """Device::copy_buffer's C++ bounds check -- an out-of-range blit copy
    must surface as a catchable Python exception, not an uncatchable
    Objective-C validation abort."""
    src = device.empty(4, np.float32)
    dst = device.empty(4, np.float32)
    nbytes = 4 * np.dtype(np.float32).itemsize
    with pytest.raises(RuntimeError):
        device._dev.copy_buffer(src._buf, 0, dst._buf, 0, nbytes + 1, True)
    with pytest.raises(RuntimeError):
        device._dev.copy_buffer(src._buf, nbytes, dst._buf, 0, nbytes, True)


def test_kernel_dispatch_works_on_private_buffer(device):
    """Pipeline.run binds buffers by GPU handle only, never touches CPU
    memory -- a Private buffer must work as a kernel operand exactly like a
    Shared one."""
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void square(
        device const float *a [[buffer(0)]],
        device       float *b [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    {
        b[id] = a[id] * a[id];
    }
    """
    pipeline = device.compile(source, "square")
    a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), storage=StorageMode.PRIVATE)
    out = device.empty(4, np.float32, storage=StorageMode.PRIVATE)
    pipeline.run([a, out], grid=4)
    np.testing.assert_array_equal(out.numpy(), [1.0, 4.0, 9.0, 16.0])


def test_texture_upload_from_private_buffer(device):
    tex = device.empty_texture((2, 4), "r32Float")
    data = np.arange(8, dtype=np.float32)
    buf = device.buffer(data, storage=StorageMode.PRIVATE)
    tex.upload_from_buffer(buf)
    np.testing.assert_array_equal(tex.numpy().reshape(-1), data)
