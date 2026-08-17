"""Correctness for advanced synchronization: Device.fence()/event()/shared_event()
and Device.queue() -- explicit GPU<->GPU and CPU<->GPU ordering beyond the
"one serial queue, commit order is enough" model the rest of the test suite
relies on (see test_async.py, test_command_buffer.py).
"""
import threading
import time

import numpy as np
import pytest

try:
    from mtlpy import shader
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


# ---------------------------------------------------------------------------
# Fence
# ---------------------------------------------------------------------------

def test_fence_orders_two_dispatches_batched_in_one_command_buffer(device):
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    fence = device.fence()
    a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    ones = device.buffer(np.ones(3, dtype=np.float32))
    mid = device.empty(3, np.float32)
    out = device.empty(3, np.float32)

    with device.command_buffer() as cb:
        add_pipeline.run([a, ones, mid], 3, cb=cb, signal_fences=[fence])
        add_pipeline.run([mid, ones, out], 3, cb=cb, wait_fences=[fence])

    np.testing.assert_allclose(out.contents, [3.0, 4.0, 5.0])


def test_fence_orders_two_self_contained_dispatches_on_same_queue(device):
    # No cb= here -- both dispatches are self-contained (their own command
    # buffer each), on this Device's own default queue. wait=False on the
    # producer means nothing forces ordering except the fence itself (no
    # CommandBuffer batching to rely on, unlike the test above).
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    fence = device.fence()
    a = device.buffer(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    ones = device.buffer(np.ones(3, dtype=np.float32))
    mid = device.empty(3, np.float32)
    out = device.empty(3, np.float32)

    add_pipeline.run([a, ones, mid], 3, wait=False, signal_fences=[fence])
    add_pipeline.run([mid, ones, out], 3, wait=True, wait_fences=[fence])

    np.testing.assert_allclose(out.contents, [12.0, 22.0, 32.0])


# ---------------------------------------------------------------------------
# Queue + Event (GPU-side only)
# ---------------------------------------------------------------------------

def test_queue_alone_runs_dispatches_correctly(device):
    """A secondary Queue works as a plain submission target, independent of
    any synchronization primitive -- the baseline multi-queue smoke test."""
    q = device.queue()
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(4, dtype=np.float32))
    b = device.buffer(np.full(4, 2.0, dtype=np.float32))
    out = device.empty(4, np.float32)

    with device.command_buffer(queue=q) as cb:
        add_pipeline.run([a, b, out], 4, cb=cb)

    np.testing.assert_allclose(out.contents, [3.0, 3.0, 3.0, 3.0])


def test_event_orders_producer_and_consumer_across_two_queues(device):
    """The producer's CommandBuffer runs on q1 and signals event=1 once its
    dispatch completes; the consumer's CommandBuffer runs on q2 and encodes a
    wait for event>=1 before its own dispatch. Without that wait, q2's
    dispatch could legitimately start before q1's producer has written mid
    (they're on independent queues -- no automatic ordering between them,
    unlike two command buffers on the *same* queue). commit(wait=True) on
    the consumer side is enough to prove the whole chain completed -- the
    result is only correct if the GPU actually honored the event dependency.
    """
    q1 = device.queue()
    q2 = device.queue()
    event = device.event()
    add_pipeline = device.compile(shader.add_kernel("float"), "add")

    a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = device.buffer(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    mid = device.empty(4, np.float32)

    with device.command_buffer(queue=q1) as cb1:
        add_pipeline.run([a, b, mid], 4, cb=cb1)
        cb1.signal_event(event, 1)

    ones = device.buffer(np.ones(4, dtype=np.float32))
    out = device.empty(4, np.float32)
    with device.command_buffer(queue=q2) as cb2:
        cb2.wait_for_event(event, 1)
        add_pipeline.run([mid, ones, out], 4, cb=cb2)

    np.testing.assert_allclose(out.contents, [12.0, 23.0, 34.0, 45.0])


def test_event_wait_for_event_before_any_dispatch(device):
    """wait_for_event() as the very first thing encoded into a fresh
    CommandBuffer (no prior encode() call) -- exercises the branch where
    CommandBuffer's shared encoder doesn't exist yet when the wait is
    spliced in."""
    q1 = device.queue()
    q2 = device.queue()
    event = device.event()
    add_pipeline = device.compile(shader.add_kernel("float"), "add")

    a = device.buffer(np.array([5.0], dtype=np.float32))
    b = device.buffer(np.array([1.0], dtype=np.float32))
    produced = device.empty(1, np.float32)

    with device.command_buffer(queue=q1) as cb1:
        add_pipeline.run([a, b, produced], 1, cb=cb1)
        cb1.signal_event(event, 7)

    out = device.empty(1, np.float32)
    cb2 = device.command_buffer(queue=q2)
    cb2.wait_for_event(event, 7)  # nothing encoded into cb2 yet
    add_pipeline.run([produced, b, out], 1, cb=cb2)
    cb2.commit()

    assert out.contents[0] == 7.0


def test_wait_for_event_after_commit_raises(device):
    cb = device.command_buffer()
    event = device.event()
    cb.commit()
    with pytest.raises(RuntimeError):
        cb.wait_for_event(event, 1)


def test_signal_event_after_commit_raises(device):
    cb = device.command_buffer()
    event = device.event()
    cb.commit()
    with pytest.raises(RuntimeError):
        cb.signal_event(event, 1)


# ---------------------------------------------------------------------------
# SharedEvent (adds CPU-visible signal/wait + cross-process handle)
# ---------------------------------------------------------------------------

def test_shared_event_cpu_signal_and_wait_roundtrip(device):
    event = device.shared_event()
    assert event.signaled_value == 0
    event.signal(3)
    assert event.signaled_value == 3
    assert event.wait(3, timeout_ms=1000) is True


def test_shared_event_wait_times_out_when_never_signaled(device):
    event = device.shared_event()
    start = time.perf_counter()
    assert event.wait(1, timeout_ms=200) is False
    assert time.perf_counter() - start >= 0.15  # actually blocked, didn't fast-path


def test_shared_event_gpu_signal_then_cpu_wait(device):
    """GPU signals via CommandBuffer.signal_event(); the CPU blocks in
    SharedEvent.wait() until that GPU-side signal fires."""
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    event = device.shared_event()
    a = device.buffer(np.ones(4, dtype=np.float32))
    b = device.buffer(np.ones(4, dtype=np.float32))
    out = device.empty(4, np.float32)

    with device.command_buffer() as cb:
        add_pipeline.run([a, b, out], 4, cb=cb)
        cb.signal_event(event, 1)

    assert event.wait(1, timeout_ms=5000) is True
    np.testing.assert_allclose(out.contents, [2.0, 2.0, 2.0, 2.0])


def test_shared_event_cpu_signal_unblocks_gpu_wait(device):
    """CPU signals via SharedEvent.signal(); a GPU-side dispatch encoded
    behind CommandBuffer.wait_for_event() only runs once that happens. The
    producer command buffer is committed wait=False (so committing alone
    can't be what makes this pass) and the CPU-side signal is delayed on a
    background thread -- the consumer's own commit(wait=True) is the only
    thing forcing the whole chain to finish, and it can only do that
    correctly if the GPU actually waited for the event."""
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    event = device.shared_event()
    a = device.buffer(np.array([100.0], dtype=np.float32))
    b = device.buffer(np.array([1.0], dtype=np.float32))
    out = device.empty(1, np.float32)

    cb = device.command_buffer()
    cb.wait_for_event(event, 1)
    add_pipeline.run([a, b, out], 1, cb=cb)
    cb.commit(wait=False)

    def signal_after_delay():
        time.sleep(0.05)
        event.signal(1)

    t = threading.Thread(target=signal_after_delay)
    t.start()
    try:
        # Forces the CPU to block until the GPU dispatch (which is itself
        # blocked on the event) completes -- there's no other synchronization
        # here, so this only returns 101.0 if the event wait actually worked.
        zero = device.buffer(np.zeros(1, dtype=np.float32))
        result = device.empty(1, np.float32)
        add_pipeline.run([out, zero, result], 1, wait=True)
        # add_pipeline.run above ran on the default queue while out's
        # producing cb ran on... the same default queue (no queue= given),
        # so its own FIFO commit-order guarantee (see test_async.py) is
        # already enough to order it after cb -- but only once cb's
        # encoded event-wait itself unblocks, which is what's actually
        # under test: without a correct wait_for_event, this read could
        # race the still-blocked producer.
        np.testing.assert_allclose(result.contents, [101.0])
    finally:
        t.join()


def test_wait_and_signal_event_interleaved_with_dispatches_in_one_command_buffer(device):
    """encode() lazily (re)opens CommandBuffer's shared compute encoder --
    wait_for_event()/signal_event() must close whatever encoder is currently
    open (Metal only allows encodeWait/encodeSignalEvent between encoders,
    not while one is active) and let a later Pipeline.run(cb=cb) call
    transparently open a fresh one. This chains three dispatches through two
    such splices in a single CommandBuffer/command buffer."""
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    event = device.event()
    a = device.buffer(np.array([1.0], dtype=np.float32))
    b = device.buffer(np.array([1.0], dtype=np.float32))
    mid1 = device.empty(1, np.float32)
    mid2 = device.empty(1, np.float32)
    out = device.empty(1, np.float32)

    cb = device.command_buffer()
    add_pipeline.run([a, b, mid1], 1, cb=cb)     # opens encoder #1
    cb.wait_for_event(event, 0)                   # closes it (value 0 is trivially satisfied)
    add_pipeline.run([mid1, b, mid2], 1, cb=cb)  # opens encoder #2
    cb.signal_event(event, 1)                      # closes it
    add_pipeline.run([mid2, b, out], 1, cb=cb)   # opens encoder #3
    cb.commit()

    assert out.contents[0] == 4.0


def test_command_buffer_dropped_without_commit_after_wait_for_event(device):
    """Regression guard for close_encoder(): once wait_for_event() has
    ended+released+nulled the shared encoder, CommandBuffer's destructor
    must not try to end/release it again (it checks `if (encoder_)`, which
    close_encoder() already nulled out) if the CommandBuffer is dropped
    without ever calling commit()."""
    import gc

    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    event = device.event()
    a = device.buffer(np.array([1.0], dtype=np.float32))
    b = device.buffer(np.array([1.0], dtype=np.float32))
    out = device.empty(1, np.float32)

    cb = device.command_buffer()
    add_pipeline.run([a, b, out], 1, cb=cb)
    cb.wait_for_event(event, 1)
    del cb
    gc.collect()  # must not crash


def test_shared_event_export_import_handle_share_state(device):
    """SharedEvent.export_handle()/Device.import_shared_event() reconstruct
    a SharedEvent backed by the *same* underlying MTL::SharedEvent -- this
    only exercises the same-process path (mtlpy doesn't provide an IPC
    channel, see SharedEventHandle's docstring), but that's exactly the
    primitive a real cross-process caller would build on top of."""
    original = device.shared_event()
    handle = original.export_handle()
    imported = device.import_shared_event(handle)

    imported.signal(42)
    assert original.signaled_value == 42

    original.signal(43)
    assert imported.signaled_value == 43
