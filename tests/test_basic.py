import numpy as np
import pytest

try:
    import mtlpy
    from mtlpy import Device, operators
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


@pytest.fixture
def device():
    return Device()


def test_buffer_roundtrip(device):
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf = device.buffer(data)
    np.testing.assert_array_equal(buf.contents, data)


def test_add(device):
    a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = device.buffer(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_array_almost_equal((a + b).contents, [5.0, 7.0, 9.0])


def test_sub(device):
    a = device.buffer(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    b = device.buffer(np.array([1.0,  2.0,  3.0],  dtype=np.float32))
    np.testing.assert_array_almost_equal((a - b).contents, [9.0, 18.0, 27.0])


def test_mul(device):
    a = device.buffer(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    b = device.buffer(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_almost_equal((a * b).contents, [4.0, 6.0, 8.0])


def test_sqrt(device):
    a = device.buffer(np.array([4.0, 9.0, 16.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(operators.sqrt(a).contents, [2.0, 3.0, 4.0])


def test_integer_types(device):
    a = device.buffer(np.array([10, 20, 30], dtype=np.int32))
    b = device.buffer(np.array([1,  2,  3],  dtype=np.int32))
    np.testing.assert_array_equal((a + b).contents, [11, 22, 33])


def test_astype(device):
    a = device.buffer(np.array([1, 2, 3], dtype=np.int32))
    b = a.astype(np.float32)
    assert b.dtype == np.float32
    np.testing.assert_array_almost_equal(b.contents, [1.0, 2.0, 3.0])


def test_custom_shader(device):
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
    a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = device.empty(4, np.float32)
    pipeline.run([a, b], 4)
    np.testing.assert_array_almost_equal(b.contents, [1.0, 4.0, 9.0, 16.0])


def test_pipeline_cache(device):
    a = device.buffer(np.ones(1000, dtype=np.float32))
    b = device.buffer(np.ones(1000, dtype=np.float32))
    for _ in range(5):
        c = a + b
    np.testing.assert_array_almost_equal(c.contents, np.full(1000, 2.0))


def test_large_buffer(device):
    n = 1_000_000
    a = device.buffer(np.ones(n, dtype=np.float32))
    b = device.buffer(np.ones(n, dtype=np.float32) * 2)
    c = a + b
    assert np.all(c.contents == 3.0)
