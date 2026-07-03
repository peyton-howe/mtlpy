"""Dispatch a kernel repeatedly in a hot loop without reallocating buffers.

Buffer.contents is a live NumPy view over the same underlying Metal
allocation (see src/mtlpy/buffer.py) -- writing buf.contents[:] = ... updates
GPU-visible memory in place, and reading it back after a wait=True dispatch
needs no reallocation either. Compile the pipeline and allocate buffers once,
outside the loop.

Note this bypasses the convenience operators (a + b, operators.sqrt(a), ...):
those allocate a brand-new output Buffer on every call, which is fine for
one-off use but wasteful in a loop like this one.
"""
import numpy as np
import mtlpy

device = mtlpy.Device()

source = """
#include <metal_stdlib>
using namespace metal;
kernel void square(
    device const float *a [[buffer(0)]],
    device       float *b [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    b[id] = a[id] * a[id];
}
"""
pipeline = device.compile(source, "square")

a   = device.buffer(np.zeros(4, dtype=np.float32))  # allocated once
out = device.empty(4, np.float32)                    # allocated once

for step in range(5):
    a.contents[:] = np.array([step, step + 1, step + 2, step + 3], dtype=np.float32)
    pipeline.run([a, out], grid=4)  # wait=True by default
    print(f"step {step}: a={a.contents}  a^2={out.contents}")
