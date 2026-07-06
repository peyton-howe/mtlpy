"""Regression coverage for the Pipeline::run() autorelease-pool /
commandBufferWithUnretainedReferences() change and PipelineCache's on-disk
binary archive: both touch object lifetime in ways a single dispatch won't
exercise. A bug here would show up as a crash, not a wrong value, so these
tests exist mainly to run at all without segfaulting.
"""
import gc
import threading

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


def test_concurrent_dispatch_from_multiple_threads(device):
    """Pipeline.run releases the GIL for its whole duration (see
    py::call_guard<gil_scoped_release>() in bindings.cpp), so multiple
    Python threads can now genuinely execute Metal calls concurrently on the
    same Device -- exercise that this is actually safe, not just
    theoretically fine (MTLCommandQueue is documented thread-safe;
    PipelineCache serializes compilation with its own mutex)."""
    errors = []

    def worker(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(50):
                n = int(rng.integers(1, 2048))
                a_np = rng.random(n, dtype=np.float32)
                b_np = rng.random(n, dtype=np.float32)
                a = device.buffer(a_np)
                b = device.buffer(b_np)
                c = a + b
                np.testing.assert_allclose(c.contents, a_np + b_np, rtol=1e-5, atol=1e-6)
        except Exception as e:  # noqa: BLE001 -- collect and re-raise on the main thread
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"{len(errors)} worker thread(s) raised: {errors}"


def test_concurrent_pipeline_compilation(device):
    """Stress PipelineCache::get_or_create's mutex specifically: many
    threads compiling distinct kernels at the same time."""
    errors = []

    def worker(i):
        try:
            source = f"""
#include <metal_stdlib>
using namespace metal;
kernel void scale_{i}(
    device const float *a [[buffer(0)]],
    device       float *b [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{{
    b[id] = a[id] * {float(i + 1)};
}}
"""
            pipeline = device.compile(source, f"scale_{i}")
            a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            a = device.buffer(a_np)
            b = device.empty(3, np.float32)
            pipeline.run([a, b], 3)
            np.testing.assert_allclose(b.contents, a_np * (i + 1))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"{len(errors)} worker thread(s) raised: {errors}"
