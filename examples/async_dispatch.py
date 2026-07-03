"""Batch several GPU dispatches without stalling on each one.

Pipeline.run(..., wait=False) commits work and returns immediately. Since
Metal retires command buffers on a queue in commit order, a later wait=True
dispatch that reads the earlier results is enough to guarantee they're ready
-- you don't need to wait after every single dispatch.
"""
import numpy as np
import mtlpy
from mtlpy import shader

device = mtlpy.Device()

n = 1_000_000
add = device.compile(shader.add_kernel("float"), "add")
ones = device.buffer(np.ones(n, dtype=np.float32))

acc = device.buffer(np.zeros(n, dtype=np.float32))
for _ in range(10):
    nxt = device.empty(n, np.float32)
    add.run([acc, ones, nxt], n, wait=False)  # fire-and-forget
    acc = nxt

zero = device.buffer(np.zeros(n, dtype=np.float32))
result = device.empty(n, np.float32)
add.run([acc, zero, result], n, wait=True)  # forces everything above to finish

print("expected all-10s, got:", result.contents[:5], "...")
