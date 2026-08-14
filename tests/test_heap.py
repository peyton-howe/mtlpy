"""Device.heap() (see src/mtlpy/heap.py): a memory pool that Buffers/
Textures are sub-allocated from instead of each getting its own standalone
MTLDevice allocation. Every resource on a Heap shares the Heap's own
storage mode -- these tests cover buffer/texture sub-allocation, that
storage mode propagates correctly (including the non-Shared staging path),
heap size/used_size bookkeeping, and heap exhaustion raising cleanly.
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

_MB = 1024 * 1024


def _heap_or_skip(device, size_bytes, storage):
    """Managed heaps (unlike a standalone Managed Buffer) require the Mac2
    GPU family -- unsupported on Apple silicon's unified memory. Device.heap()
    raises a clean RuntimeError for that case (see csrc/heap.cpp); treat it
    as an expected hardware limitation, not a test failure."""
    try:
        return device.heap(size_bytes, storage=storage)
    except RuntimeError as e:
        if storage == StorageMode.MANAGED and "not supported on this GPU" in str(e):
            pytest.skip("Managed heaps not supported on this GPU family")
        raise


def test_heap_reports_at_least_requested_size(device):
    heap = device.heap(_MB)
    assert heap.size >= _MB
    assert heap.used_size == 0


def test_heap_default_storage_is_shared(device):
    heap = device.heap(_MB)
    assert heap.storage == StorageMode.SHARED


@pytest.mark.parametrize("storage", list(StorageMode))
def test_heap_buffer_from_int_has_heap_storage(device, storage):
    heap = _heap_or_skip(device, _MB, storage)
    buf = heap.empty(64, np.float32)
    assert buf.storage == storage


@pytest.mark.parametrize("storage", list(StorageMode))
def test_heap_buffer_from_ndarray_roundtrips(device, storage):
    heap = _heap_or_skip(device, _MB, storage)
    data = np.arange(16, dtype=np.float32)
    buf = heap.buffer(data)
    assert buf.storage == storage
    np.testing.assert_array_equal(buf.numpy(), data)


def test_heap_managed_unsupported_raises_cleanly(device):
    """Whether or not this GPU supports Managed heaps, the outcome must be
    a catchable Python exception, never a process abort (see csrc/heap.cpp's
    guard against MTLHeapDescriptor's own hard-abort validation failure)."""
    try:
        device.heap(_MB, storage=StorageMode.MANAGED)
    except RuntimeError:
        pass  # expected on a GPU family without Managed heap support


def test_heap_used_size_grows_with_allocations(device):
    heap = device.heap(_MB)
    assert heap.used_size == 0
    buf = heap.empty(1024, np.float32)  # kept alive -- see used_size's docstring
    assert heap.used_size > 0
    del buf


def test_heap_buffer_usable_in_kernel_dispatch(device):
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
    heap = device.heap(_MB, storage=StorageMode.PRIVATE)
    a   = heap.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    out = heap.empty(4, np.float32)
    pipeline.run([a, out], grid=4)
    np.testing.assert_array_equal(out.numpy(), [1.0, 4.0, 9.0, 16.0])


def test_heap_texture_roundtrips(device):
    heap = device.heap(_MB)
    tex = heap.empty_texture((2, 4), "r32Float")
    assert tex.shape == (2, 4)
    data = np.arange(8, dtype=np.float32).reshape(2, 4)
    tex.upload(data)
    np.testing.assert_array_equal(tex.numpy(), data)


def test_heap_private_texture_is_private(device):
    heap = device.heap(_MB, storage=StorageMode.PRIVATE)
    tex = heap.empty_texture((2, 4), "r32Float")
    assert tex.is_private


def test_heap_exhaustion_raises(device):
    heap = device.heap(4096)  # small, deliberately easy to exhaust
    with pytest.raises(RuntimeError):
        # A buffer far larger than the whole heap must fail cleanly, not
        # abort the process.
        heap.empty(_MB * 64, np.float32)


def test_heap_repr(device):
    heap = device.heap(_MB, storage=StorageMode.PRIVATE)
    r = repr(heap)
    assert "storage=private" in r
