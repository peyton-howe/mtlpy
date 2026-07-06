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


def _scalar(name: str, op: str, t: str, reflected: bool = False) -> str:
    lhs, rhs = ("scalar", "a[id]") if reflected else ("a[id]", "scalar")
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void {name}(
    device const {t} *a [[buffer(0)]],
    constant     {t} &scalar [[buffer(1)]],
    device       {t} *c [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{{
    c[id] = {lhs} {op} {rhs};
}}
"""


def _compare(name: str, op: str, t: str) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void {name}(
    device const {t}  *a [[buffer(0)]],
    device const {t}  *b [[buffer(1)]],
    device       bool *c [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{{
    c[id] = a[id] {op} b[id];
}}
"""


def _compare_scalar(name: str, op: str, t: str) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void {name}(
    device const {t}  *a [[buffer(0)]],
    constant     {t}  &scalar [[buffer(1)]],
    device       bool *c [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{{
    c[id] = a[id] {op} scalar;
}}
"""


def _reduce_pair(name: str, combine: str, t: str) -> str:
    """One pass of a multi-pass tree reduction: thread `id` combines input
    elements 2*id and 2*id+1 into output[id]. For an odd leftover element
    (no pair available), it passes the lone element through unchanged --
    this works for any associative op (sum/max/min) without needing a
    per-dtype identity value (0 for sum, -inf for max, etc.)."""
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void {name}(
    device const {t}  *in  [[buffer(0)]],
    device       {t}  *out [[buffer(1)]],
    constant     uint &n   [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{{
    uint i = id * 2;
    if (i + 1 < n) {{
        out[id] = {combine};
    }} else {{
        out[id] = in[i];
    }}
}}
"""


def _negate(t: str) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void negate(
    device const {t} *a [[buffer(0)]],
    device       {t} *b [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{{
    b[id] = -a[id];
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
def div_kernel(t: str) -> str:  return _binary("div",  "/",    t)
def sqrt_kernel(t: str) -> str: return _unary("sqrt_op", "sqrt", t)
def cos_kernel(t: str) -> str:  return _unary("cos_op",  "cos",  t)
def sin_kernel(t: str) -> str:  return _unary("sin_op",  "sin",  t)
def tan_kernel(t: str) -> str:  return _unary("tan_op",  "tan",  t)
def exp_kernel(t: str) -> str:  return _unary("exp_op",  "exp",  t)
def log_kernel(t: str) -> str:  return _unary("log_op",  "log",  t)
def cast_kernel(src: str, dst: str) -> str: return _cast(src, dst)

def add_scalar_kernel(t: str) -> str:  return _scalar("add_scalar",  "+", t)
def sub_scalar_kernel(t: str) -> str:  return _scalar("sub_scalar",  "-", t)
def rsub_scalar_kernel(t: str) -> str: return _scalar("rsub_scalar", "-", t, reflected=True)
def mul_scalar_kernel(t: str) -> str:  return _scalar("mul_scalar",  "*", t)
def div_scalar_kernel(t: str) -> str:  return _scalar("div_scalar",  "/", t)
def rdiv_scalar_kernel(t: str) -> str: return _scalar("rdiv_scalar", "/", t, reflected=True)
def negate_kernel(t: str) -> str:      return _negate(t)

def eq_kernel(t: str) -> str: return _compare("eq", "==", t)
def ne_kernel(t: str) -> str: return _compare("ne", "!=", t)
def lt_kernel(t: str) -> str: return _compare("lt", "<",  t)
def le_kernel(t: str) -> str: return _compare("le", "<=", t)
def gt_kernel(t: str) -> str: return _compare("gt", ">",  t)
def ge_kernel(t: str) -> str: return _compare("ge", ">=", t)

def eq_scalar_kernel(t: str) -> str: return _compare_scalar("eq_scalar", "==", t)
def ne_scalar_kernel(t: str) -> str: return _compare_scalar("ne_scalar", "!=", t)
def lt_scalar_kernel(t: str) -> str: return _compare_scalar("lt_scalar", "<",  t)
def le_scalar_kernel(t: str) -> str: return _compare_scalar("le_scalar", "<=", t)
def gt_scalar_kernel(t: str) -> str: return _compare_scalar("gt_scalar", ">",  t)
def ge_scalar_kernel(t: str) -> str: return _compare_scalar("ge_scalar", ">=", t)

def reduce_sum_kernel(t: str) -> str: return _reduce_pair("reduce_sum", "in[i] + in[i + 1]",         t)
def reduce_max_kernel(t: str) -> str: return _reduce_pair("reduce_max", "max(in[i], in[i + 1])", t)
def reduce_min_kernel(t: str) -> str: return _reduce_pair("reduce_min", "min(in[i], in[i + 1])", t)
