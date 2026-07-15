"""Buffer.shape, .reshape(), .numpy(), and __array__.

Buffer.contents is deliberately always flat (see
test_basic.py::test_buffer_from_multidim_array) -- these tests cover the
shape-aware surface layered on top of that without disturbing it.
"""
import numpy as np
import pytest

try:
    from mtlpy import Device
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_shape_from_ndarray(device):
    img = np.arange(24, dtype=np.float32).reshape(4, 6)
    buf = device.buffer(img)
    assert buf.shape == (4, 6)
    assert buf.size == 24
    np.testing.assert_array_equal(buf.contents, img.reshape(-1))  # still flat


def test_shape_defaults_to_flat_for_1d(device):
    buf = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert buf.shape == (3,)


def test_empty_accepts_shape_tuple(device):
    buf = device.empty((2, 3, 4), np.float32)
    assert buf.shape == (2, 3, 4)
    assert buf.size == 24


def test_numpy_returns_shaped_view(device):
    img = np.arange(24, dtype=np.float32).reshape(4, 6)
    buf = device.buffer(img)
    out = buf.numpy()
    assert out.shape == (4, 6)
    np.testing.assert_array_equal(out, img)


def test_numpy_is_zero_copy_view(device):
    buf = device.buffer(np.arange(6, dtype=np.float32).reshape(2, 3))
    view = buf.numpy()
    view[0, 0] = 99.0
    np.testing.assert_array_equal(buf.contents[:1], [99.0])  # write-through


def test_array_protocol(device):
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    buf = device.buffer(img)
    np.testing.assert_array_equal(np.asarray(buf), img)
    np.testing.assert_array_equal(np.array(buf), img)


def test_array_protocol_dtype_conversion(device):
    buf = device.buffer(np.array([1, 2, 3], dtype=np.int32))
    out = np.array(buf, dtype=np.float32)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_reshape_shares_underlying_buffer(device):
    flat = device.buffer(np.arange(12, dtype=np.float32))
    grid = flat.reshape(3, 4)

    assert grid.shape == (3, 4)
    assert grid._buf is flat._buf  # same Metal allocation, no realloc
    np.testing.assert_array_equal(grid.numpy(), np.arange(12).reshape(3, 4))


def test_reshape_accepts_tuple_or_varargs(device):
    flat = device.buffer(np.arange(12, dtype=np.float32))
    assert flat.reshape(3, 4).shape == (3, 4)
    assert flat.reshape((3, 4)).shape == (3, 4)


def test_reshape_rejects_mismatched_size(device):
    flat = device.buffer(np.arange(12, dtype=np.float32))
    with pytest.raises(ValueError):
        flat.reshape(3, 5)


def test_elementwise_op_preserves_shape(device):
    a = device.buffer(np.arange(6, dtype=np.float32).reshape(2, 3))
    b = device.buffer(np.ones((2, 3), dtype=np.float32))
    c = a + b
    assert c.shape == (2, 3)
    np.testing.assert_array_equal(c.numpy(), a.numpy() + 1.0)


def test_elementwise_op_allows_mismatched_shape_same_size(device):
    # A Metal buffer has no shape of its own -- .shape is Python-side
    # metadata layered on top, and it's .size (flat element count) that
    # actually has to match for two buffers to be valid operands.
    a = device.buffer(np.arange(24, dtype=np.float32).reshape(4, 6))
    b = device.buffer(np.arange(24, dtype=np.float32))  # same size, different shape
    c = a + b
    assert c.shape == a.shape  # output takes the first operand's shape
    np.testing.assert_array_equal(c.numpy(), (a.numpy() + b.numpy().reshape(4, 6)))


def test_elementwise_op_rejects_mismatched_size(device):
    a = device.buffer(np.arange(4, dtype=np.float32))
    b = device.buffer(np.arange(6, dtype=np.float32))
    with pytest.raises(ValueError):
        a + b


def test_astype_preserves_shape(device):
    a = device.buffer(np.arange(6, dtype=np.int32).reshape(2, 3))
    b = a.astype(np.float32)
    assert b.shape == (2, 3)
