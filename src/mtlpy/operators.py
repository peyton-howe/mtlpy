from __future__ import annotations
import numpy as np
from . import shader, utils
from .buffer import Buffer


def _unary(buf: Buffer, shader_fn, func_name: str) -> Buffer:
    metal_type = utils.to_metal(buf.dtype)
    pipeline   = buf._device.compile(shader_fn(metal_type), func_name)
    out        = buf._device.empty(buf.size, buf.dtype)
    pipeline.run([buf, out], buf.size)
    return out


def sqrt(buf: Buffer) -> Buffer: return _unary(buf, shader.sqrt_kernel, "sqrt_op")
def cos(buf: Buffer)  -> Buffer: return _unary(buf, shader.cos_kernel,  "cos_op")
def sin(buf: Buffer)  -> Buffer: return _unary(buf, shader.sin_kernel,  "sin_op")
def tan(buf: Buffer)  -> Buffer: return _unary(buf, shader.tan_kernel,  "tan_op")
def exp(buf: Buffer)  -> Buffer: return _unary(buf, shader.exp_kernel,  "exp_op")
def log(buf: Buffer)  -> Buffer: return _unary(buf, shader.log_kernel,  "log_op")


def _reduce(buf: Buffer, shader_fn, func_name: str):
    """Multi-pass tree reduction: each pass halves the element count by
    combining adjacent pairs (see shader._reduce_pair), until one element
    remains. O(log n) dispatches rather than one -- not the fastest possible
    GPU reduction (a single-pass threadgroup-shared-memory version would be),
    but simple and needs no new dispatch machinery. Accumulates in the
    buffer's own dtype, like Metal's native operators elsewhere in this
    library -- e.g. summing a large int16 buffer can overflow, matching
    what the equivalent int16 accumulator would do in a hand-written kernel.
    """
    device     = buf._device
    metal_type = utils.to_metal(buf.dtype)
    pipeline   = device.compile(shader_fn(metal_type), func_name)

    current, n = buf, buf.size
    while n > 1:
        out_n  = (n + 1) // 2
        out    = device.empty(out_n, buf.dtype)
        n_buf  = device.buffer(np.array([n], dtype=np.uint32))
        pipeline.run([current, out, n_buf], out_n)
        current, n = out, out_n
    return current.contents[0].item()


def sum(buf: Buffer):  return _reduce(buf, shader.reduce_sum_kernel, "reduce_sum")
def max(buf: Buffer):  return _reduce(buf, shader.reduce_max_kernel, "reduce_max")
def min(buf: Buffer):  return _reduce(buf, shader.reduce_min_kernel, "reduce_min")
def mean(buf: Buffer) -> float: return sum(buf) / buf.size
