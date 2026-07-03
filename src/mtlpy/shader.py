def _binary(name: str, op: str, t: str) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void {name}(
    device const {t} *a [[buffer(0)]],
    device const {t} *b [[buffer(1)]],
    device       {t} *c [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{{
    c[id] = a[id] {op} b[id];
}}
"""


def _unary(name: str, fn: str, t: str) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void {name}(
    device const {t} *a [[buffer(0)]],
    device       {t} *b [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{{
    b[id] = {fn}(a[id]);
}}
"""


def _cast(src_t: str, dst_t: str) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void cast(
    device const {src_t} *a [[buffer(0)]],
    device       {dst_t} *b [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{{
    b[id] = {dst_t}(a[id]);
}}
"""


def add_kernel(t: str) -> str:  return _binary("add",  "+",    t)
def sub_kernel(t: str) -> str:  return _binary("sub",  "-",    t)
def mul_kernel(t: str) -> str:  return _binary("mul",  "*",    t)
def sqrt_kernel(t: str) -> str: return _unary("sqrt_op", "sqrt", t)
def cos_kernel(t: str) -> str:  return _unary("cos_op",  "cos",  t)
def sin_kernel(t: str) -> str:  return _unary("sin_op",  "sin",  t)
def tan_kernel(t: str) -> str:  return _unary("tan_op",  "tan",  t)
def exp_kernel(t: str) -> str:  return _unary("exp_op",  "exp",  t)
def log_kernel(t: str) -> str:  return _unary("log_op",  "log",  t)
def cast_kernel(src: str, dst: str) -> str: return _cast(src, dst)
