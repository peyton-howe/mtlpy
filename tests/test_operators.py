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
    ("div", lambda a, b: a / b),
]
# Deliberately excludes bool: NumPy itself doesn't treat bool uniformly
# across these ops (e.g. `-` raises TypeError on bool arrays), so bool gets
# its own narrower test below instead of sharing this matrix.
BINARY_DTYPES = [
    np.float32, np.float16,
    np.int32, np.uint32, np.int16, np.uint16, np.int64, np.uint64,
]


@pytest.mark.parametrize("dtype", BINARY_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("op_name,op", BINARY_OPS, ids=[c[0] for c in BINARY_OPS])
def test_binary_op_dtypes(device, op_name, op, dtype):
    a_np = np.array([10, 20, 30, 40], dtype=dtype)
    b_np = np.array([1, 2, 3, 4], dtype=dtype)
    a = device.buffer(a_np)
    b = device.buffer(b_np)
    got = op(a, b).contents
    np.testing.assert_array_equal(got, op(a_np, b_np))


def test_binary_op_bool_add(device):
    # bool isn't in BINARY_DTYPES above because NumPy disallows `-` on bool
    # arrays outright; `+` still works (Metal treats it like int promotion).
    a_np = np.array([True, False, True], dtype=np.bool_)
    b_np = np.array([True, True, False], dtype=np.bool_)
    a = device.buffer(a_np)
    b = device.buffer(b_np)
    np.testing.assert_array_equal((a + b).contents, a_np + b_np)


SCALAR_OPS = [
    ("add",  lambda a, s: a + s,  lambda a, s: a + s),
    ("radd", lambda a, s: s + a,  lambda a, s: s + a),
    ("sub",  lambda a, s: a - s,  lambda a, s: a - s),
    ("rsub", lambda a, s: s - a,  lambda a, s: s - a),
    ("mul",  lambda a, s: a * s,  lambda a, s: a * s),
    ("rmul", lambda a, s: s * a,  lambda a, s: s * a),
    ("div",  lambda a, s: a / s,  lambda a, s: a / s),
    ("rdiv", lambda a, s: s / a,  lambda a, s: s / a),
]


@pytest.mark.parametrize("name,mtlpy_op,numpy_op", SCALAR_OPS, ids=[c[0] for c in SCALAR_OPS])
def test_scalar_broadcast(device, name, mtlpy_op, numpy_op):
    a_np = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    a = device.buffer(a_np)
    scalar = 2.0
    got = mtlpy_op(a, scalar).contents
    np.testing.assert_allclose(got, numpy_op(a_np, scalar), rtol=1e-5)


def test_negate(device):
    a_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    a = device.buffer(a_np)
    np.testing.assert_allclose((-a).contents, -a_np)


def test_integer_division_truncates_like_metal_not_numpy(device):
    """Buffer / Buffer uses Metal's native `/` for the shared dtype: true
    division for floats, but C-style truncating division for integers --
    unlike NumPy's `/`, which always promotes integers to float64. This is
    intentional (see Buffer.__truediv__'s docstring); this test pins the
    documented behavior so it isn't silently "fixed" into matching NumPy."""
    a = device.buffer(np.array([7, 8, 9], dtype=np.int32))
    b = device.buffer(np.array([2, 2, 2], dtype=np.int32))
    np.testing.assert_array_equal((a / b).contents, [3, 4, 4])


COMPARE_OPS = [
    ("eq", lambda a, b: a == b, lambda a, b: a == b),
    ("ne", lambda a, b: a != b, lambda a, b: a != b),
    ("lt", lambda a, b: a < b,  lambda a, b: a < b),
    ("le", lambda a, b: a <= b, lambda a, b: a <= b),
    ("gt", lambda a, b: a > b,  lambda a, b: a > b),
    ("ge", lambda a, b: a >= b, lambda a, b: a >= b),
]


@pytest.mark.parametrize("name,mtlpy_op,numpy_op", COMPARE_OPS, ids=[c[0] for c in COMPARE_OPS])
def test_compare_buffer_vs_buffer(device, name, mtlpy_op, numpy_op):
    a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_np = np.array([2.0, 2.0, 2.0, 5.0], dtype=np.float32)
    a, b = device.buffer(a_np), device.buffer(b_np)
    out = mtlpy_op(a, b)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out.contents, numpy_op(a_np, b_np))


@pytest.mark.parametrize("name,mtlpy_op,numpy_op", COMPARE_OPS, ids=[c[0] for c in COMPARE_OPS])
def test_compare_buffer_vs_scalar(device, name, mtlpy_op, numpy_op):
    a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    a = device.buffer(a_np)
    out = mtlpy_op(a, 2.0)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out.contents, numpy_op(a_np, 2.0))


def test_compare_reflected_scalar(device):
    # Python auto-reflects rich comparisons: `3.0 > a` tries
    # (3.0).__gt__(a) -> NotImplemented -> a.__lt__(3.0).
    a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    a = device.buffer(a_np)
    np.testing.assert_array_equal((3.0 > a).contents, 3.0 > a_np)
    np.testing.assert_array_equal((2.0 == a).contents, 2.0 == a_np)


def test_buffer_is_unhashable(device):
    # Matches NumPy's ndarray: defining __eq__ to return an array (not a
    # bool) means the natural, safe choice is to make the type unhashable.
    a = device.buffer(np.array([1.0], dtype=np.float32))
    with pytest.raises(TypeError):
        hash(a)


IPLACE_OPS = [
    ("iadd", lambda buf, v: buf.__iadd__(v), lambda a, v: a + v),
    ("isub", lambda buf, v: buf.__isub__(v), lambda a, v: a - v),
    ("imul", lambda buf, v: buf.__imul__(v), lambda a, v: a * v),
    ("itruediv", lambda buf, v: buf.__itruediv__(v), lambda a, v: a / v),
]


@pytest.mark.parametrize("name,mtlpy_op,numpy_op", IPLACE_OPS, ids=[c[0] for c in IPLACE_OPS])
def test_inplace_scalar_ops_reuse_buffer(device, name, mtlpy_op, numpy_op):
    a_np = np.array([4.0, 8.0, 12.0], dtype=np.float32)
    a = device.buffer(a_np)
    underlying = a._buf
    result = mtlpy_op(a, 2.0)
    assert result is a  # in-place ops return self
    assert a._buf is underlying  # and must not allocate a new Metal buffer
    np.testing.assert_allclose(a.contents, numpy_op(a_np, 2.0))


def test_inplace_buffer_op_reuses_buffer(device):
    a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    a, b = device.buffer(a_np), device.buffer(b_np)
    underlying = a._buf
    a += b
    assert a._buf is underlying
    np.testing.assert_allclose(a.contents, a_np + b_np)


def test_cross_device_binary_op_raises():
    from mtlpy import Device
    dev1, dev2 = Device(), Device()
    a = dev1.buffer(np.array([1.0], dtype=np.float32))
    b = dev2.buffer(np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError):
        a + b


@pytest.mark.parametrize("n", [1, 2, 3, 7, 17, 33, 1023, 1024, 1025])
def test_reductions_odd_and_pow2_sizes(device, n):
    # Exercises operators._reduce's pairwise-tree odd-leftover handling
    # across sizes that land on both sides of several halving boundaries.
    rng = np.random.default_rng(n)
    x_np = (rng.random(n, dtype=np.float32) * 10 - 5).astype(np.float32)
    x = device.buffer(x_np)
    assert operators.sum(x) == pytest.approx(float(x_np.sum()), rel=1e-3, abs=1e-2)
    assert operators.max(x) == pytest.approx(float(x_np.max()))
    assert operators.min(x) == pytest.approx(float(x_np.min()))
    assert operators.mean(x) == pytest.approx(float(x_np.mean()), rel=1e-3, abs=1e-2)


def test_reduction_integer_dtype_and_mean_is_float(device):
    a_np = np.array([10, 20, 30, 40], dtype=np.int32)
    a = device.buffer(a_np)
    assert operators.sum(a) == int(a_np.sum())
    assert isinstance(operators.sum(a), int)
    assert operators.mean(a) == pytest.approx(float(a_np.mean()))
    assert isinstance(operators.mean(a), float)


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
    # float64 has no Metal equivalent, so it's downcast to float32.
    expected_dtype = np.float32 if np.dtype(dst_dtype) == np.float64 else np.dtype(dst_dtype)
    assert out.dtype == expected_dtype
    np.testing.assert_allclose(out.contents, src.astype(expected_dtype))


def test_single_element_buffer(device):
    a = device.buffer(np.array([7.0], dtype=np.float32))
    b = device.buffer(np.array([5.0], dtype=np.float32))
    np.testing.assert_allclose((a + b).contents, [12.0])
