"""Correctness for Device.command_buffer(): batching multiple Pipeline.run()
dispatches into one MTLCommandBuffer submission instead of each paying its
own command-buffer-create + commit + waitUntilCompleted round trip.
"""
import gc

import numpy as np
import pytest

try:
    from mtlpy import shader
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_two_dispatches_batched_produce_correct_result(device):
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    ones = device.buffer(np.ones(3, dtype=np.float32))
    mid = device.empty(3, np.float32)
    out = device.empty(3, np.float32)

    with device.command_buffer() as cb:
        add_pipeline.run([a, ones, mid], 3, cb=cb)
        add_pipeline.run([mid, ones, out], 3, cb=cb)

    np.testing.assert_allclose(out.contents, [3.0, 4.0, 5.0])


def test_run_returns_zero_sentinel_while_batched(device):
    # Per-dispatch GPU timing isn't meaningful once dispatches share a
    # command buffer -- only CommandBuffer.commit()'s combined timing is.
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(3, dtype=np.float32))
    b = device.buffer(np.ones(3, dtype=np.float32))
    out = device.empty(3, np.float32)

    with device.command_buffer() as cb:
        gpu_start, gpu_end = add_pipeline.run([a, b, out], 3, cb=cb)
    assert (gpu_start, gpu_end) == (0.0, 0.0)


def test_commit_returns_gpu_timing(device):
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(3, dtype=np.float32))
    b = device.buffer(np.ones(3, dtype=np.float32))
    out = device.empty(3, np.float32)

    cb = device.command_buffer()
    add_pipeline.run([a, b, out], 3, cb=cb)
    gpu_start, gpu_end = cb.commit()
    assert gpu_end >= gpu_start >= 0.0


def test_exception_inside_with_block_discards_batch(device):
    # __exit__ deliberately does not commit if the block raised -- a
    # partially-encoded batch is discarded, not submitted.
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.array([100.0], dtype=np.float32))
    b = device.buffer(np.array([1.0], dtype=np.float32))
    out = device.buffer(np.array([-1.0], dtype=np.float32))  # sentinel value

    with pytest.raises(ValueError):
        with device.command_buffer() as cb:
            add_pipeline.run([a, b, out], 1, cb=cb)
            raise ValueError("boom")

    assert out.contents[0] == -1.0  # dispatch never actually ran


def test_double_commit_raises(device):
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(1, dtype=np.float32))
    b = device.buffer(np.ones(1, dtype=np.float32))
    out = device.empty(1, np.float32)

    cb = device.command_buffer()
    add_pipeline.run([a, b, out], 1, cb=cb)
    cb.commit()
    with pytest.raises(RuntimeError):
        cb.commit()


def test_encode_after_commit_raises(device):
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(1, dtype=np.float32))
    b = device.buffer(np.ones(1, dtype=np.float32))
    out = device.empty(1, np.float32)

    cb = device.command_buffer()
    cb.commit()
    with pytest.raises(RuntimeError):
        add_pipeline.run([a, b, out], 1, cb=cb)


def test_commit_wait_false_then_sync_read(device):
    # Same ordering guarantee test_async.py relies on for a plain wait=False
    # dispatch: Metal retires command buffers on a queue in commit order, so
    # a later wait=True dispatch on the same queue guarantees this one
    # finished first, even though we never waited on it directly.
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.array([5.0], dtype=np.float32))
    b = device.buffer(np.array([1.0], dtype=np.float32))
    out = device.empty(1, np.float32)

    cb = device.command_buffer()
    add_pipeline.run([a, b, out], 1, cb=cb)
    cb.commit(wait=False)

    zero = device.buffer(np.zeros(1, dtype=np.float32))
    result = device.empty(1, np.float32)
    add_pipeline.run([out, zero, result], 1, wait=True)

    assert result.contents[0] == 6.0


def test_input_buffer_survives_gc_between_run_and_commit(device):
    # The encode()...commit() gap is a *separate* Python call, unlike the
    # single-dispatch wait=false path where setBuffer() and commit() happen
    # back-to-back inside one C++ call with no Python code -- and thus no
    # GC -- able to run in between. This directly exercises that gap: drop
    # every Python reference to the input buffers, force a collection, and
    # only *then* commit -- Metal's own encoder-side retain (taken when
    # Pipeline.run(cb=cb) calls setBuffer) must be what's actually keeping
    # the underlying MTL::Buffer alive, not the Python wrapper object.
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    out = device.empty(3, np.float32)

    cb = device.command_buffer()

    def encode_with_temporary_inputs():
        a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = device.buffer(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        add_pipeline.run([a, b, out], 3, cb=cb)
        # a, b go out of scope here -- no other reference survives

    encode_with_temporary_inputs()
    gc.collect()  # force collection before commit(), not just rely on refcounting

    cb.commit()
    np.testing.assert_allclose(out.contents, [11.0, 22.0, 33.0])


def test_wait_with_cb_raises(device):
    # wait has no effect once dispatches are batched -- CommandBuffer.commit
    # controls waiting instead. Explicitly passing wait alongside cb must
    # raise rather than silently ignore it, so a caller who tacks cb=cb onto
    # an existing wait=True/wait=False call site notices wait stopped
    # applying instead of a semantics change going unnoticed.
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(1, dtype=np.float32))
    b = device.buffer(np.ones(1, dtype=np.float32))
    out = device.empty(1, np.float32)

    with device.command_buffer() as cb:
        with pytest.raises(ValueError):
            add_pipeline.run([a, b, out], 1, wait=True, cb=cb)
        with pytest.raises(ValueError):
            add_pipeline.run([a, b, out], 1, wait=False, cb=cb)
    # cb=None (the default) is unaffected, and wait keeps its old meaning
    add_pipeline.run([a, b, out], 1, wait=True)


def test_failed_dispatch_poisons_batch_even_without_context_manager(device):
    # commit() itself (not just __exit__) must refuse to submit a batch a
    # prior dispatch failed partway into -- this is the bare
    # `cb = device.command_buffer()` pattern the README documents as
    # equally supported, so it needs the same "a failed dispatch discards
    # the whole batch" guarantee the context-manager path gets for free.
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(1, dtype=np.float32))
    b = device.buffer(np.ones(1, dtype=np.float32))
    out = device.buffer(np.array([-1.0], dtype=np.float32))  # sentinel

    cb = device.command_buffer()
    add_pipeline.run([a, b, out], 1, cb=cb)  # succeeds, gets encoded
    with pytest.raises(RuntimeError):
        # missing the 3rd (params) buffer this kernel doesn't need, but
        # simulate a failed dispatch the straightforward way: too few
        # buffers for a kernel that requires more than 2.
        bad_pipeline = device.compile(
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void needs_three(device const float* a [[buffer(0)]], "
            "device const float* b [[buffer(1)]], device const float* c [[buffer(2)]], "
            "device float* out [[buffer(3)]], uint id [[thread_position_in_grid]]) "
            "{ out[id] = a[id] + b[id] + c[id]; }",
            "needs_three",
        )
        bad_pipeline.run([a, b], 1, cb=cb)  # too few buffers -- raises

    # Even though the first dispatch encoded fine, the batch is now poisoned.
    with pytest.raises(RuntimeError):
        cb.commit()
    assert out.contents[0] == -1.0  # nothing was ever submitted


def test_many_command_buffers_mixed_committed_and_gced(device):
    # Regression test: an earlier implementation used a per-CommandBuffer
    # NSAutoreleasePool, which crashed (pool-nesting violation / "Command
    # encoder released without endEncoding") once more than one
    # CommandBuffer was live and they weren't destroyed in strict creation
    # order -- which Python's garbage collector gives no guarantee of.
    # Deliberately leaves some uncommitted to exercise the destructor-only
    # cleanup path (encoder created but commit() never called).
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    live = []
    for i in range(20):
        cb = device.command_buffer()
        a = device.buffer(np.array([float(i)], dtype=np.float32))
        b = device.buffer(np.array([1.0], dtype=np.float32))
        out = device.empty(1, np.float32)
        add_pipeline.run([a, b, out], 1, cb=cb)
        if i % 2 == 0:
            cb.commit()
        else:
            live.append(cb)  # keep some alive out of creation order
    del live
