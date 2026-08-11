"""Correctness for Pipeline.run(wait=False).

Reading a Buffer's contents right after an async (wait=False) dispatch would
race the GPU -- there's no Python-level fence to wait on independently. These
tests instead force ordering by committing a second, wait=True dispatch on
the same (serial) command queue that consumes the async dispatch's output:
Metal retires command buffers on a queue in commit order, so waiting on the
later one guarantees the earlier one has already completed.
"""
import time

import numpy as np
import pytest

try:
    from mtlpy import shader
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_async_dispatch_then_sync_read(device):
    a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = device.buffer(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    c = device.empty(4, np.float32)

    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    add_pipeline.run([a, b, c], 4, wait=False)

    zero = device.buffer(np.zeros(4, dtype=np.float32))
    d = device.empty(4, np.float32)
    add_pipeline.run([c, zero, d], 4, wait=True)

    np.testing.assert_allclose(d.contents, [11.0, 22.0, 33.0, 44.0])


def test_async_batch_accumulate(device):
    """Chain many wait=False dispatches (the realistic reason to use async:
    avoid stalling per-dispatch), then force a sync at the end."""
    n = 256
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    ones = device.buffer(np.ones(n, dtype=np.float32))

    acc = device.buffer(np.zeros(n, dtype=np.float32))
    for _ in range(50):
        nxt = device.empty(n, np.float32)
        add_pipeline.run([acc, ones, nxt], n, wait=False)
        acc = nxt

    zero = device.buffer(np.zeros(n, dtype=np.float32))
    result = device.empty(n, np.float32)
    add_pipeline.run([acc, zero, result], n, wait=True)

    np.testing.assert_allclose(result.contents, np.full(n, 50.0))


def test_async_dispatch_stays_correct_with_cpu_work_before_sync(device):
    """Regression test for the pattern documented in the README's 'Batching
    dispatches' section as an alternative to CommandBuffer batching: a
    wait=False dispatch submits immediately, so real CPU-side work can run
    concurrently with it before a later wait=True dispatch syncs. This only
    tests correctness (the FIFO commit-order guarantee must still hold no
    matter how much CPU work happens before the sync point, not just
    immediately after) -- the performance characteristics (measured ~1.2x
    faster than CommandBuffer batching in this scenario) are a benchmarks/
    concern, not something to assert on timing here."""
    a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = device.buffer(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    c = device.empty(4, np.float32)

    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    add_pipeline.run([a, b, c], 4, wait=False)

    # Simulate real CPU-side work happening before the sync point (e.g.
    # preparing the next dispatch's arguments) -- the FIFO ordering
    # guarantee must hold regardless of how long this takes.
    t_end = time.perf_counter() + 0.01
    while time.perf_counter() < t_end:
        pass

    zero = device.buffer(np.zeros(4, dtype=np.float32))
    d = device.empty(4, np.float32)
    add_pipeline.run([c, zero, d], 4, wait=True)

    np.testing.assert_allclose(d.contents, [11.0, 22.0, 33.0, 44.0])
