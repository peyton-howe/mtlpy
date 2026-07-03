from __future__ import annotations
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
