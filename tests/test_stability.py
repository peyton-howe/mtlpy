"""Regression coverage for the Pipeline::run() autorelease-pool /
commandBufferWithUnretainedReferences() change and PipelineCache's on-disk
binary archive: both touch object lifetime in ways a single dispatch won't
exercise. A bug here would show up as a crash, not a wrong value, so these
tests exist mainly to run at all without segfaulting.
"""
import gc

import numpy as np
import pytest

try:
    from mtlpy import Device
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_repeated_dispatch_does_not_crash_or_corrupt(device):
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(1, 4096))
        a_np = rng.random(n, dtype=np.float32)
        b_np = rng.random(n, dtype=np.float32)
        a = device.buffer(a_np)
        b = device.buffer(b_np)
        c = a + b
        np.testing.assert_allclose(c.contents, a_np + b_np, rtol=1e-5, atol=1e-6)


def test_many_short_lived_devices():
    """Each Device owns its own PipelineCache, which opens (and, on
    teardown, serializes to) the same on-disk binary archive path -- make
    sure repeatedly creating/destroying Devices doesn't corrupt it."""
    for _ in range(25):
        dev = Device()
        a = dev.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = dev.buffer(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        np.testing.assert_allclose((a + b).contents, [5.0, 7.0, 9.0])
        del dev  # triggers PipelineCache::~PipelineCache() -> archive serialize


def test_buffer_outlives_local_python_refs(device):
    """Buffer.contents returns a view holding an _mtlpy_buf backref so the
    Buffer (and therefore the Device) stays alive as long as the array does
    -- exercise that under GC pressure."""
    def make_view():
        buf = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        return buf.contents  # buf itself goes out of scope here

    view = make_view()
    gc.collect()
    np.testing.assert_allclose(view, [1.0, 2.0, 3.0])
