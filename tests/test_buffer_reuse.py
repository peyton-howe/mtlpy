"""Buffer.contents is a live view over the same Metal allocation (see
src/mtlpy/buffer.py), so a hot loop can allocate buffers once and reuse them
via in-place writes/reads instead of reallocating every iteration. This
pins down that pattern: no new allocation, correct results across many
iterations of write-dispatch-read.
"""
import numpy as np
import pytest

try:
    import mtlpy  # noqa: F401
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")

_SQUARE_SOURCE = """
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


def test_in_place_write_updates_same_underlying_buffer(device):
    a = device.buffer(np.zeros(4, dtype=np.float32))
    original_ptr = a._buf.data_ptr

    a.contents[:] = [1.0, 2.0, 3.0, 4.0]

    assert a._buf.data_ptr == original_ptr
    np.testing.assert_array_equal(a.contents, [1.0, 2.0, 3.0, 4.0])


def test_repeated_dispatch_reuses_buffers_without_reallocating(device):
    pipeline = device.compile(_SQUARE_SOURCE, "square")

    a   = device.buffer(np.zeros(4, dtype=np.float32))
    out = device.empty(4, np.float32)
    a_ptr, out_ptr = a._buf.data_ptr, out._buf.data_ptr

    for step in range(50):
        values = np.array([step, step + 1, step + 2, step + 3], dtype=np.float32)
        a.contents[:] = values
        pipeline.run([a, out], grid=4)

        # Same underlying Metal buffers the whole loop -- no reallocation.
        assert a._buf.data_ptr == a_ptr
        assert out._buf.data_ptr == out_ptr
        np.testing.assert_allclose(out.contents, values ** 2)


def test_contents_view_is_stable_across_accesses(device):
    """Each .contents access builds a fresh ndarray, but it must point at
    the same underlying memory -- otherwise reuse across loop iterations
    wouldn't actually avoid allocation."""
    a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    first = a.contents
    second = a.contents
    assert first.ctypes.data == second.ctypes.data
