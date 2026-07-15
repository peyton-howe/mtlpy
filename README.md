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
  texture.{h,cpp}      MTL::Texture wrapper (1D/2D/3D)
  sampler.{h,cpp}      MTL::SamplerState wrapper
  pipeline.{h,cpp}     Dispatches a compiled MTL::ComputePipelineState
  command_buffer.{h,cpp}  Batches multiple Pipeline::run() dispatches into
                          one MTL::CommandBuffer submission
  pipeline_cache.{h,cpp}  Compiles-once cache, keyed on (source, function name),
                          backed by an on-disk MTL::BinaryArchive
  metal_impl.mm        Single Obj-C++ translation unit providing the
                        NS::/CA::/MTL:: private implementations
  bindings.cpp         pybind11 module definition (`_mtlpy`)
src/mtlpy/          Python package (src layout, for PyPI)
  device.py            Device: buffer/empty/texture/sampler/compile, list_devices(), wraps _mtlpy.Device
  buffer.py             Buffer: NumPy-backed contents, arithmetic/comparison/in-place operators
  texture.py             Texture, Sampler: wrap _mtlpy.Texture/_mtlpy.Sampler
  pipeline.py           Pipeline, CommandBuffer: thin wrappers over _mtlpy.Pipeline/CommandBuffer
  operators.py          sqrt/cos/sin/tan/exp/log, sum/max/min/mean reductions
  shader.py             Generates Metal Shading Language source per dtype/texture type
  utils.py              NumPy dtype <-> Metal type/pixel format mapping
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
  `Pipeline.run` validates the buffer/texture/sampler counts against the
  kernel's own argument reflection, so passing too few of any of them raises
  a clear Python exception instead of leaving a Metal argument unbound
  (undefined behavior).
- **Textures**: 1D/2D/3D `Texture`s (`device.texture(array, pixel_format)` /
  `device.empty_texture(shape, pixel_format)`) and `Sampler`s for kernels
  written against `texture2d<...>`/etc. rather than raw buffers -- see
  [Textures](#textures) below.
- **Shapes**: `Buffer.shape` tracks the logical shape a buffer was created
  or `reshape()`d with (elementwise ops preserve it); `Buffer.numpy()` /
  `np.asarray(buf)` return contents in that shape. `Buffer.contents` itself
  stays flat regardless — see [Shapes and NumPy interop](#shapes-and-numpy-interop).
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
- **Batched dispatches**: `Device.command_buffer()` batches multiple
  `Pipeline.run()` calls into one `MTLCommandBuffer` submission instead of
  one per dispatch -- see [Batching dispatches](#batching-dispatches) below.
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

## Shapes and NumPy interop

`Buffer.contents` is deliberately always flat (see
[Custom kernels](#custom-kernels) and the note above on unified memory) — but
every `Buffer` also tracks a logical `.shape`, set when you create it from an
ndarray or via `device.empty(shape, dtype)`, and preserved by the elementwise
operators:

```python
img = np.arange(24, dtype=np.float32).reshape(4, 6)
buf = device.buffer(img)

buf.shape          # (4, 6)
buf.contents.shape # (24,)  -- always flat
buf.numpy().shape   # (4, 6) -- contents reshaped to buf.shape, still zero-copy

np.asarray(buf)     # same as buf.numpy() -- Buffer implements __array__
np.array(buf, dtype=np.float64)  # dtype conversion via the same protocol

grid = buf.reshape(2, 12)  # new Buffer, same underlying Metal allocation
```

`.numpy()` and `__array__` are both zero-copy views, same as `.contents` —
reshaping a flat contiguous array is always a view in NumPy, never a copy, so
none of this allocates or duplicates GPU memory. `.reshape()` similarly
shares the same `MTL::Buffer` rather than reallocating.

Elementwise operators (`+`, `-`, `*`, comparisons, ...) check `.size`, not
`.shape` — a Metal buffer has no shape of its own (it's just bytes), so two
buffers with equal flat size but different declared `.shape` are still valid
operands; the result takes the first operand's `.shape`. `.shape` is
purely Python-side bookkeeping layered on top, not something Metal itself
knows about.

## Textures

`Texture` wraps `MTL::Texture` (1D/2D/3D) for kernels that want
`texture2d<...>`-style access instead of raw `device float*` buffers --
useful for image-processing-style kernels, and for sampling (bilinear
filtering, addressing modes) rather than plain indexing.

```python
img = np.random.rand(64, 64).astype(np.float32)          # (height, width)
tex = device.texture(img, "r32Float")                      # uploads in one call

invert_source = """
#include <metal_stdlib>
using namespace metal;
kernel void invert(
    texture2d<float, access::read_write> tex [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float4 c = tex.read(gid);
    tex.write(1.0 - c, gid);
}
"""
pipeline = device.compile(invert_source, "invert")
pipeline.run([], grid=(64, 64, 1), textures=[tex])   # no buffers, one texture

result = tex.download()  # or np.asarray(tex) / tex.numpy()
```

`Pipeline.run` takes `buffers`, `textures`, and `samplers` as separate lists
because Metal Shading Language gives each its own independent binding
namespace (`[[buffer(n)]]` / `[[texture(n)]]` / `[[sampler(n)]]`) -- list
position `i` binds to index `i` in that namespace, same convention buffers
already use. A sampling kernel needs a `Sampler` too:

```python
sample_source = """
#include <metal_stdlib>
using namespace metal;
kernel void downscale(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    sampler                          smp [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    dst.write(src.sample(smp, uv), gid);
}
"""
small = device.empty_texture((32, 32), "r32Float")
smp   = device.sampler(linear=True)   # linear=False for nearest-neighbor
downscale = device.compile(sample_source, "downscale")
downscale.run([], grid=(32, 32, 1), textures=[tex, small], samplers=[smp])
```

A few things that differ from `Buffer`:

- **No zero-copy `.contents`.** Metal doesn't guarantee a texture's
  CPU-visible memory is a tightly packed array the way a Shared-storage
  `Buffer`'s is (rows can be padded/tiled internally), so there's no
  `Buffer.contents` equivalent. `.upload()`/`.download()` (and `.numpy()`/
  `__array__`, which call `.download()`) are genuine copies via
  `MTL::Texture`'s `replaceRegion`/`getBytes`.
- **`shape` excludes the channel count on input, includes it on output.**
  `device.empty_texture(shape, pixel_format)` takes a *spatial* shape only
  -- `(width,)`, `(height, width)`, or `(depth, height, width)` -- with
  channel count implied by `pixel_format` (`"rgba8Unorm"` is 4-channel,
  `"r32Float"` is 1-channel). `Texture.shape` (and what `.download()`
  returns) appends a trailing channel dim when `channels > 1`, matching
  common image-array conventions. `empty_texture` raises if a 3-element
  `shape` looks like it still has that channel axis attached (its last
  element equals the format's channel count) -- if you have an `(H, W, C)`
  array, either use `device.texture(data, pixel_format)` (which strips it
  for you) or pass `shape[:-1]`.
- **Pixel formats, not dtypes.** A small, deliberately non-exhaustive set
  covering 8-bit and float image data (Metal defines 100+ pixel formats,
  most for graphics rather than compute): `r8Unorm`, `rgba8Unorm`,
  `r16Float`, `rgba16Float`, `r32Float`, `rgba32Float`, `r32Uint`,
  `rgba32Uint` (`src/mtlpy/utils.py`). `Unorm` formats store small integers
  but kernels read/write them as `float` in `[0, 1]` -- Metal normalizes
  automatically; `shader.texture_type(dims, msl_scalar_type, access)`
  generates the right MSL type string (`texture2d<float, access::sample>`,
  etc.) if you don't want to hand-write it.

### Moving data in and out of a Texture

`device.texture(data, pixel_format)` and `.upload()`/`.download()` (above)
are the simple default path -- always available, but CPU-side copies
(`replaceRegion`/`getBytes`) that only work on the default `Shared` storage
mode. There's a faster, GPU-side path for every CPU/Buffer/Texture
direction, each suited to a different data source and each working
regardless of storage mode (including `private=True`, see below) -- see
`benchmarks/README.md` for the measurements behind these:

| Direction | Method | Mechanism | Notes |
|---|---|---|---|
| CPU -> Texture | `tex.upload(data)` | CPU `replaceRegion` | Simple default; `Shared` storage only |
| CPU -> Texture | `tex.upload_fast(data)` | CPU->`Buffer` memcpy + GPU blit | Up to ~9x faster at 4K; works on `private=True` |
| `Buffer` -> Texture | `tex.upload_from_buffer(buf)` | GPU blit | Same mechanism `upload_fast` uses, without the implicit staging `Buffer` -- use this directly if you already have a `Buffer` (e.g. reusing one across a hot loop instead of allocating fresh each call) |
| Texture -> CPU | `tex.download()` / `.numpy()` / `np.asarray(tex)` | CPU `getBytes` | Simple default; `Shared` storage only |
| Texture -> CPU | `tex.download_fast()` | GPU compute kernel + `Buffer`->CPU (zero-copy) | ~1.5-1.6x faster at 1080p/4K; works on `private=True`; raises `NotImplementedError` on `Unorm` formats |
| Texture -> `Buffer` | `tex.to_buffer()` | GPU compute kernel | What `download_fast` calls before `.numpy()` -- use this directly if you want the `Buffer`, not a numpy array (e.g. feeding it straight into another kernel) |
| Texture -> Texture | `src.copy_to(dst)` | GPU blit | Works on any pixel format (including `Unorm`) and any `Shared`/`Private` combination -- e.g. copying a `Shared` texture you populated with `.upload()` into a `Private` one before a hot compute loop |

`empty_texture(..., readable=, writable=, private=)` controls usage flags
and storage mode at creation. `private=True` (`MTLStorageModePrivate`,
GPU-only memory) requires the GPU-side methods above -- `.upload()`/
`.download()` raise a clear error on a private texture, since Metal itself
rejects `replaceRegion`/`getBytes` on `Private` storage.

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

## Batching dispatches

Each `Pipeline.run()` call submits its own `MTLCommandBuffer` by default —
one command-buffer-create + commit + (if `wait=True`) `waitUntilCompleted()`
round trip per dispatch. For a fixed sequence of dispatches that always run
together (a multi-pass kernel, or any "run these N things then read the
result" pattern), `Device.command_buffer()` batches them into one submission
instead, as a context manager:

```python
with device.command_buffer() as cb:
    horizontal_pass.run([], grid, textures=[src, mid], cb=cb)
    vertical_pass.run([], grid, textures=[mid, dst], cb=cb)
# one submit, one wait, covering both dispatches
```

`Pipeline.run(..., cb=cb)` encodes into `cb`'s shared encoder instead of
committing its own command buffer — `wait` is ignored in that case (you
can't partially wait on part of a not-yet-committed command buffer), and it
always returns `(0.0, 0.0)`: per-dispatch GPU timing isn't meaningful once
dispatches share a command buffer, only `cb.commit()`'s combined timing is.
The `with` block commits (and waits, by default) on normal exit; if the
block raises, it does *not* commit — a partially-encoded batch is discarded
rather than submitted, the same way a database transaction rolls back on
exception instead of committing a partial write. Measured ~2x faster than
two separate `wait=True` dispatches for a two-pass texture kernel.

Without a context manager: `cb = device.command_buffer()`, encode dispatches
into it the same way, then `cb.commit(wait=True)` (the default) or
`cb.commit(wait=False)` to defer waiting the same way `Pipeline.run(...,
wait=False)` does — a later `wait=True` dispatch on the same queue still
guarantees this batch finished first (Metal retires command buffers on a
queue in commit order). Calling `.commit()` more than once, or encoding into
a `CommandBuffer` after it's committed, raises `RuntimeError`.

**`CommandBuffer` vs. a plain `wait=False` chain** (see [Async
dispatch](#features) above): both avoid stalling between dependent
dispatches, but they're not interchangeable —

- Back-to-back dispatches with no CPU-side work between them: roughly tied
  either way.
- CPU-side work between dispatches (e.g. computing the next dispatch's
  arguments): a `wait=False` chain wins, measured ~1.2x faster for a 4K
  two-pass kernel with 2ms of CPU work in between. `wait=False` *submits*
  immediately, so the GPU starts executing that dispatch while the CPU is
  still busy; `CommandBuffer` batching defers *all* submission until
  `commit()`, so the GPU sits idle until the whole batch has been encoded.
- Sequence not known upfront (the next dispatch depends on inspecting
  something first): only a `wait=False` chain fits -- nothing in a
  `CommandBuffer` batch executes until it's fully encoded and committed, so
  you can't make encoding decisions based on an earlier batched dispatch's
  result without breaking the batch anyway.

`CommandBuffer` is the better fit for a fixed, known-upfront sequence with
little CPU work in between (its original motivating case: a multi-pass
kernel). A `wait=False` chain remains the right tool when there's real work
to overlap with GPU execution, or the sequence is decided dynamically.

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
- `test_shapes.py` — `.shape`, `.reshape()`, `.numpy()`, and `__array__`,
  including that `.contents` stays flat and elementwise ops preserve shape.
- `test_texture.py` — texture creation/shape across pixel formats and
  dimensionalities, upload/download roundtrips, a read-write compute kernel,
  and a sampling kernel (multi-texture + sampler binding).
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
