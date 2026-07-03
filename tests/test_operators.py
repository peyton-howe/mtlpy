"""Correctness coverage for all operators/dtypes, not just the sqrt/add/sub/mul
smoke tests in test_basic.py."""
import numpy as np
import pytest

try:
    from mtlpy import operators
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def _unary_cases():
    return [
        ("sqrt", operators.sqrt, np.sqrt, np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)),
        ("cos",  operators.cos,  np.cos,  np.linspace(-3.0, 3.0, 32, dtype=np.float32)),
        ("sin",  operators.sin,  np.sin,  np.linspace(-3.0, 3.0, 32, dtype=np.float32)),
        ("tan",  operators.tan,  np.tan,  np.linspace(-1.0, 1.0, 32, dtype=np.float32)),
        ("exp",  operators.exp,  np.exp,  np.linspace(-2.0, 2.0, 32, dtype=np.float32)),
        ("log",  operators.log,  np.log,  np.linspace(0.1, 10.0, 32, dtype=np.float32)),
    ]


@pytest.mark.parametrize(
    "mtlpy_fn,numpy_fn,data",
    [c[1:] for c in _unary_cases()],
    ids=[c[0] for c in _unary_cases()],
)
def test_unary_op(device, mtlpy_fn, numpy_fn, data):
    buf = device.buffer(data)
    got = mtlpy_fn(buf).contents
    np.testing.assert_allclose(got, numpy_fn(data), rtol=1e-4, atol=1e-5)


BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
]
BINARY_DTYPES = [np.float32, np.float16, np.int32, np.uint32, np.int16]


@pytest.mark.parametrize("dtype", BINARY_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("op_name,op", BINARY_OPS, ids=[c[0] for c in BINARY_OPS])
def test_binary_op_dtypes(device, op_name, op, dtype):
    a_np = np.array([10, 20, 30, 40], dtype=dtype)
    b_np = np.array([1, 2, 3, 4], dtype=dtype)
    a = device.buffer(a_np)
    b = device.buffer(b_np)
    got = op(a, b).contents
    np.testing.assert_array_equal(got, op(a_np, b_np))


def test_mismatched_size_raises(device):
    a = device.buffer(np.ones(4, dtype=np.float32))
    b = device.buffer(np.ones(5, dtype=np.float32))
    with pytest.raises(ValueError):
        a + b


def test_mismatched_dtype_raises(device):
    a = device.buffer(np.ones(4, dtype=np.float32))
    b = device.buffer(np.ones(4, dtype=np.int32))
    with pytest.raises(TypeError):
        a + b


ASTYPE_PAIRS = [
    (np.int32, np.float32),
    (np.float32, np.int32),
    (np.float32, np.float64),
    (np.int16, np.int32),
]


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    ASTYPE_PAIRS,
    ids=[f"{np.dtype(s).name}-to-{np.dtype(d).name}" for s, d in ASTYPE_PAIRS],
)
def test_astype_pairs(device, src_dtype, dst_dtype):
    src = np.array([1, 2, 3, 4], dtype=src_dtype)
    buf = device.buffer(src)
    out = buf.astype(dst_dtype)
    assert out.dtype == np.dtype(dst_dtype)
    np.testing.assert_allclose(out.contents, src.astype(dst_dtype))


def test_single_element_buffer(device):
    a = device.buffer(np.array([7.0], dtype=np.float32))
    b = device.buffer(np.array([5.0], dtype=np.float32))
    np.testing.assert_allclose((a + b).contents, [12.0])
