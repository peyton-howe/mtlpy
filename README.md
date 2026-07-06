# mtlpy

Python bindings for GPU compute on Apple Metal, built on [pybind11](https://github.com/pybind/pybind11)
and Apple's [metal-cpp](https://developer.apple.com/metal/cpp/). Write a Metal
compute kernel as a string, dispatch it over NumPy arrays, get the result back
as a NumPy array — no separate build step, no manual buffer plumbing.

```python
import numpy as np
import mtlpy

device = mtlpy.Device()
a = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
b = device.buffer(np.array([10.0, 20.0, 30.0], dtype=np.float32))

print((a + b).contents)  # [11. 22. 33.]
```

## Why this exists

mtlpy is a from-scratch rewrite of the`metalgpu`
project. Rather than build on top of that codebase's ctypes-based bindings
and global singleton state, mtlpy starts over with a few deliberate
improvements:

- **pybind11 instead of ctypes** — real type safety across the Python/C++
  boundary, and Metal errors propagate as Python exceptions instead of
  silent failures.
- **No global singleton state** — each `Device` owns its own command queue
  and pipeline cache; nothing is saved/restored behind your back.
- **Pipeline compile caching** — a compute pipeline is compiled once per
  (shader source, function name) and reused, both within a process and
  (via an on-disk Metal binary archive) across process launches.
- **Async dispatch** — `Pipeline.run(..., wait=False)` lets you batch work
  without stalling on every call.

## Status

Alpha, but built, tested, and benchmarked on real Apple Silicon hardware —
see [Building from source](#building-from-source) and the test suite for
current coverage.

## Architecture

```
metal-cpp/          Apple's C++ Metal headers (git submodule)
csrc/                C++ extension (pybind11 + metal-cpp)
  device.{h,cpp}       MTL::Device + MTL::CommandQueue owner
  buffer.{h,cpp}       MTL::Buffer wrapper (shared-storage, CPU/GPU unified memory)
  pipeline.{h,cpp}     Dispatches a compiled MTL::ComputePipelineState
  pipeline_cache.{h,cpp}  Compiles-once cache, keyed on (source, function name),
                          backed by an on-disk MTL::BinaryArchive
  metal_impl.mm        Single Obj-C++ translation unit providing the
                        NS::/CA::/MTL:: private implementations
  bindings.cpp         pybind11 module definition (`_mtlpy`)
src/mtlpy/          Python package (src layout, for PyPI)
  device.py            Device: buffer/empty/compile, list_devices(), wraps _mtlpy.Device
  buffer.py             Buffer: NumPy-backed contents, arithmetic/comparison/in-place operators
  pipeline.py           Pipeline: thin wrapper over _mtlpy.Pipeline
  operators.py          sqrt/cos/sin/tan/exp/log, sum/max/min/mean reductions
  shader.py             Generates Metal Shading Language source per dtype
  utils.py              NumPy dtype <-> Metal type name mapping
tests/               pytest suite
benchmarks/          Standalone performance baseline scripts
examples/            Runnable usage examples
```

Each `Device` in Python owns exactly one `MTL::Device`, one `MTL::CommandQueue`,
and one `PipelineCache`. Buffers use `MTL::ResourceStorageModeShared`, so on
Apple Silicon's unified memory there's no copy between CPU and GPU views of
the same allocation — `Buffer.contents` is a NumPy array backed directly by
GPU-visible memory (accessing `.contents` is a true zero-copy view; writing
new data into it via `buf.contents[:] = arr` is still a real memcpy from
`arr`'s own memory, same as it would be for any destination).

## Features

- **Elementwise operators**: `+`, `-`, `*`, `/`, unary `-`, and in-place
  `+=`/`-=`/`*=`/`/=` (which dispatch in-place, into the same `Buffer`, with
  no extra allocation) on `Buffer` — each also works with a NumPy/Python
  scalar on either side (`buf + 5.0`, `5.0 - buf`), not just `Buffer op
  Buffer`. Plus `sqrt`, `cos`, `sin`, `tan`, `exp`, `log`, and `astype` for
  dtype conversion.
- **Comparisons**: `==`, `!=`, `<`, `<=`, `>`, `>=` (against another `Buffer`
  or a scalar) return a `bool` `Buffer`, matching NumPy's `ndarray`
  convention — which also makes `Buffer` unhashable, same tradeoff NumPy
  makes.
- **Reductions**: `operators.sum`/`max`/`min`/`mean` — an O(log n) multi-pass
  tree reduction returning a plain Python scalar.
- **Custom kernels**: compile and dispatch arbitrary Metal Shading Language
  source directly (see [Custom kernels](#custom-kernels) below).
  `Pipeline.run` validates the buffer count against the kernel's own
  argument reflection, so passing too few buffers raises a clear Python
  exception instead of leaving a Metal buffer argument unbound (undefined
  behavior).
- **Dtype support**: `float32`, `float16`, `int32`, `uint32`,
  `int16`, `uint16`, `int64`, `uint64`, `bool` — mapped to their Metal
  equivalents (`float`, `half`, `int`, `uint`, `short`, `ushort`,
  `long`, `ulong`, `bool`) in `src/mtlpy/utils.py`. `float64` has no Metal
  equivalent (no Apple GPU supports double precision), so it's silently
  downcast to `float32` at buffer creation. Note that `Buffer / Buffer` uses
  Metal's native `/` for the shared dtype (truncating for integers), not
  NumPy's always-promote-to-float64 semantics.
- **Pipeline caching**: identical (source, function name) pairs are compiled
  once per process and reused; a binary archive on disk
  (`~/Library/Caches/mtlpy/pipelines.metallib`) carries compiled pipelines
  across process launches too. `Device.flush_cache()` (or using `Device` as
  a context manager: `with mtlpy.Device() as d:`) serializes it on demand,
  rather than only when the `Device` is garbage collected.
- **Async dispatch**: `wait=False` commits work without blocking; Metal
  retires command buffers on a queue in commit order, so a later `wait=True`
  dispatch that reads the result is enough to synchronize (see
  `examples/async_dispatch.py`). `Pipeline.run` releases the GIL for the
  whole call, so other Python threads keep running during the GPU wait
  instead of being blocked for its full duration.
- **Multi-GPU support**: `mtlpy.list_devices()` lists every Metal-capable GPU
  on the machine; `mtlpy.Device(index=...)` selects one (the default targets
  the system default GPU).
- **Errors as exceptions**: shader compile failures, missing kernel
  functions, mismatched buffer counts, mismatched-`Device` operands, and GPU
  execution errors all raise Python exceptions with a clear message, instead
  of failing silently or invoking undefined behavior.

## Building from source

Requires macOS with Metal support, Xcode (for the Metal/Objective-C++
toolchain), CMake, and Python 3.9+.

```bash
git clone --recursive git@github.com:peyton-howe/mtlpy.git
cd mtlpy
pip install -e ".[dev]"
```

If you already cloned without `--recursive`:

```bash
git submodule update --init
```

## Quick start

```python
import numpy as np
import mtlpy

device = mtlpy.Device()

a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
b = device.buffer(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))

c = a + b
print(c.contents)          # numpy.ndarray([11. 22. 33. 44.])

d = mtlpy.operators.sqrt(a)
print(d.contents)

e = a.astype(np.int32)
print(e.dtype, e.contents)
```

## Custom kernels

`Device.compile(source, function_name)` compiles arbitrary Metal Shading
Language and returns a `Pipeline` you can dispatch directly:

```python
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

a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
b = device.empty(4, np.float32)
pipeline.run([a, b], grid=4)

print(b.contents)  # [1. 4. 9. 16.]
```

`grid` may be an int (1D dispatch) or a 3-tuple/list for 2D/3D dispatch.
Threadgroup sizing is computed automatically from the pipeline's
`thread_execution_width` and `max_threads_per_threadgroup`.

## Reusing buffers in a hot loop

`Buffer.contents` is a live NumPy view over the same underlying Metal
allocation, not a copy — writing `buf.contents[:] = ...` updates GPU-visible
memory in place, and reading it back after a `wait=True` dispatch needs no
reallocation either. For a kernel dispatched repeatedly (e.g. in a `while`
loop), compile the pipeline and allocate buffers once, then just write/read
`.contents` each iteration:

```python
pipeline = device.compile(source, "square")

a   = device.buffer(np.zeros(4, dtype=np.float32))  # allocated once
out = device.empty(4, np.float32)                    # allocated once

while running:
    a.contents[:] = get_next_input()   # in-place write, no realloc
    pipeline.run([a, out], grid=4)     # wait=True by default
    consume(out.contents)              # in-place read, no realloc
```

The out-of-place convenience operators (`a + b`, `operators.sqrt(a)`,
`astype`, etc.) don't follow this pattern — each call allocates a fresh
output `Buffer` internally, which is fine for one-off use but wasteful in a
tight loop. The in-place operators (`a += b`, `a *= 2.0`, ...) do reuse `a`'s
own buffer with no extra allocation, if that fits your loop. See
`examples/reuse_buffers.py`.

## Testing

```bash
pytest tests/
```

- `test_basic.py` / `test_operators.py` — correctness for every operator
  (arithmetic, scalar broadcasting, comparisons, in-place, reductions),
  dtype, and `astype` conversion, plus error handling for mismatched buffer
  sizes/dtypes/devices and wrong kernel argument counts.
- `test_async.py` — `wait=False` dispatch ordering.
- `test_buffer_reuse.py` — in-place `.contents` writes and repeated dispatch
  against the same buffers, without reallocation.
- `test_stability.py` — repeated-dispatch and object-lifetime stress tests
  (regression coverage for the Metal object-ownership rules in `csrc/`),
  plus multi-threaded dispatch/compilation tests (`Pipeline.run` releases
  the GIL, so this exercises genuinely concurrent Metal calls).
- `test_pipeline_persistence.py` — spawns separate processes to verify the
  on-disk pipeline binary archive is actually written and read back, and
  that `Device.flush_cache()` writes it on demand.

## Benchmarking

```bash
python benchmarks/bench.py
```

Measures first-dispatch (compile-included) and steady-state warm-dispatch
latency/throughput for every operator across a range of buffer sizes, with
NumPy CPU timings alongside for context. Each run is saved as JSON
(timestamped, tagged with the git commit) under `benchmarks/results/` so you
can baseline future changes:

```bash
python benchmarks/bench.py --baseline benchmarks/results/<earlier-run>.json
```

`benchmarks/demosaic_bench.py` is a separate, more involved benchmark
comparing the edge-aware Bayer demosaicing kernel
(`benchmarks/bayer2rgb_ea_kernel.txt`) against OpenCV's own
`COLOR_Bayer*2BGR_EA` (requires `pip install -e ".[bench]"`), covering both
single-shot dispatch latency and realistic streaming throughput (a rotating
buffer pool pipelining dispatches instead of waiting on every frame).

## License

MIT
