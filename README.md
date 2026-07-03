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

Alpha. The scaffolding — C++ extension, Python package, build config, tests,
and benchmarks — is in place but has not yet been built or run: that
requires a Mac with Xcode, which this was developed without. If you're
picking this up on a Mac, see [Building from source](#building-from-source)
and start with the test suite.

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
  device.py            Device: buffer/empty/compile, wraps _mtlpy.Device
  buffer.py             Buffer: NumPy-backed contents, arithmetic operators
  pipeline.py           Pipeline: thin wrapper over _mtlpy.Pipeline
  operators.py          sqrt/cos/sin/tan/exp/log
  shader.py             Generates Metal Shading Language source per dtype
  utils.py              NumPy dtype <-> Metal type name mapping
tests/               pytest suite
benchmarks/          Standalone performance baseline script
examples/            Runnable usage examples
```

Each `Device` in Python owns exactly one `MTL::Device`, one `MTL::CommandQueue`,
and one `PipelineCache`. Buffers use `MTL::ResourceStorageModeShared`, so on
Apple Silicon's unified memory there's no copy between CPU and GPU views of
the same allocation — `Buffer.contents` is a NumPy array backed directly by
GPU-visible memory.

## Features

- **Elementwise operators**: `+`, `-`, `*` on `Buffer`, plus `sqrt`, `cos`,
  `sin`, `tan`, `exp`, `log`, and `astype` for dtype conversion.
- **Custom kernels**: compile and dispatch arbitrary Metal Shading Language
  source directly (see [Custom kernels](#custom-kernels) below).
- **Dtype support**: `float32`, `float64`, `float16`, `int32`, `uint32`,
  `int16`, `uint16`, `int64`, `uint64`, `bool` — mapped to their Metal
  equivalents (`float`, `double`, `half`, `int`, `uint`, `short`, `ushort`,
  `long`, `ulong`, `bool`) in `src/mtlpy/utils.py`.
- **Pipeline caching**: identical (source, function name) pairs are compiled
  once per process and reused; a binary archive on disk
  (`~/Library/Caches/mtlpy/pipelines.metallib`) carries compiled pipelines
  across process launches too.
- **Async dispatch**: `wait=False` commits work without blocking; Metal
  retires command buffers on a queue in commit order, so a later `wait=True`
  dispatch that reads the result is enough to synchronize (see
  `examples/async_dispatch.py`).
- **Errors as exceptions**: shader compile failures, missing kernel
  functions, and GPU execution errors all raise Python exceptions with
  Metal's own error text, instead of failing silently.

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

The convenience operators (`a + b`, `operators.sqrt(a)`, `astype`, etc.)
don't follow this pattern — each call allocates a fresh output `Buffer`
internally, which is fine for one-off use but wasteful in a tight loop. See
`examples/reuse_buffers.py`.

## Testing

```bash
pytest tests/
```

- `test_basic.py` / `test_operators.py` — correctness for every operator,
  dtype, and `astype` conversion, plus error handling for mismatched
  buffer sizes/dtypes.
- `test_async.py` — `wait=False` dispatch ordering.
- `test_buffer_reuse.py` — in-place `.contents` writes and repeated dispatch
  against the same buffers, without reallocation.
- `test_stability.py` — repeated-dispatch and object-lifetime stress tests
  (regression coverage for the Metal object-ownership rules in `csrc/`).
- `test_pipeline_persistence.py` — spawns separate processes to verify the
  on-disk pipeline binary archive is actually written and read back.

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

## License

MIT
