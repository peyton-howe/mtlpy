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


def test_device_freed_deterministically_after_texture_upload():
    """Device._staging_buffer() (backing Texture.upload()/.download()'s
    defaults -- see Device.texture()) lazily caches a Heap as
    Device._staging_heap. Heap holds a reference back to its owning Device
    (needed for staging non-Shared writes) -- if that were a plain strong
    reference, it would close a cycle (Device -> _staging_heap -> Heap ->
    Device) that CPython's refcounting alone can never break, contradicting
    this library's usual instant-on-last-ref teardown (see e.g.
    Heap.used_size's docstring) and deferring the underlying MTL::Device's
    release to whenever CPython's cyclic collector happens to run -- which,
    with gc disabled below, is never. This is a *weakref* import away from
    always failing outside a test that happens to trigger a gc pass."""
    import gc
    import weakref

    from mtlpy import Device

    gc.disable()
    try:
        dev = Device()
        tex = dev.texture(np.zeros((4, 4), dtype=np.float32), "r32Float")
        del tex  # only Device._staging_heap references anything now

        ref = weakref.ref(dev)
        del dev
        assert ref() is None, (
            "Device was not freed by plain refcounting alone after a texture "
            "upload -- Heap._device (or similar) is holding a strong reference "
            "back to Device, closing a reference cycle"
        )
    finally:
        gc.enable()


def test_device_freed_deterministically_after_buffer_from_texture():
    """Same issue as test_device_freed_deterministically_after_texture_upload,
    for Device._texture_to_buffer_pipelines (backing buffer_from_texture()):
    a cached Pipeline holding a strong reference back to its owning Device
    would close a Device -> that cache -> Pipeline -> Device cycle the same
    way."""
    import gc
    import weakref

    from mtlpy import Device

    gc.disable()
    try:
        dev = Device()
        tex = dev.texture(np.zeros((4, 4), dtype=np.float32), "r32Float")
        out = dev.buffer_from_texture(tex)
        del tex, out

        ref = weakref.ref(dev)
        del dev
        assert ref() is None, (
            "Device was not freed by plain refcounting alone after "
            "buffer_from_texture() -- Pipeline._device (or similar) is "
            "holding a strong reference back to Device, closing a reference cycle"
        )
    finally:
        gc.enable()


def test_heap_from_public_api_keeps_its_device_alive():
    """The flip side of the two tests above: Device._staging_heap needed a
    *weak* back-reference to avoid a cycle, but a Heap from the public
    Device.heap() must still hold its Device *strongly* -- Device never
    caches those back (no cycle risk), and every other resource in this
    library (Buffer, Texture, ...) already keeps its owning Device alive
    for as long as it's held. A factory function that returns only the Heap
    (not the Device that created it) is a realistic pattern -- confirm it
    still works instead of the Device being freed out from under the Heap
    (which previously surfaced as buf._device silently becoming None, then
    an unrelated AttributeError far from the real cause)."""
    from mtlpy import Device
    from mtlpy.utils import StorageMode

    def make_heap():
        dev = Device()
        return dev.heap(4096, storage=StorageMode.SHARED)

    heap = make_heap()  # the local `dev` above is now out of scope
    buf = heap.buffer(np.ones(4, dtype=np.float32))
    assert buf._device is not None
    result = buf + buf  # touches ._device via _binary_op's cross-device check
    np.testing.assert_allclose(result.contents, [2.0, 2.0, 2.0, 2.0])


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
