# mtlpy

Python bindings for GPU compute on Apple Metal, built on [nanobind](https://github.com/wjakob/nanobind)
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

- **nanobind instead of ctypes** — real type safety across the Python/C++
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
csrc/                C++ extension (nanobind + metal-cpp)
  device.{h,cpp}       MTL::Device + MTL::CommandQueue owner
  buffer.{h,cpp}       MTL::Buffer wrapper (shared-storage, CPU/GPU unified memory)
  texture.{h,cpp}      MTL::Texture wrapper (1D/2D/3D)
  sampler.{h,cpp}      MTL::SamplerState wrapper
  pipeline.{h,cpp}     Dispatches a compiled MTL::ComputePipelineState
  command_buffer.{h,cpp}  Batches multiple Pipeline::run() dispatches into
                          one MTL::CommandBuffer submission
  pipeline_cache.{h,cpp}  Compiles-once cache, keyed on (source, function name),
                          backed by an on-disk MTL::BinaryArchive
  queue.{h,cpp}        A secondary MTL::CommandQueue (Device.queue())
  event.{h,cpp}        MTL::Event/MTL::SharedEvent wrappers (Device.event()/.shared_event())
  fence.{h,cpp}        MTL::Fence wrapper (Device.fence())
  metal_impl.mm        Single Obj-C++ translation unit providing the
                        NS::/CA::/MTL:: private implementations
  bindings.cpp         nanobind module definition (`_mtlpy`)
src/mtlpy/          Python package (src layout, for PyPI)
  device.py            Device: buffer/empty/texture/sampler/compile, list_devices(), wraps _mtlpy.Device
  buffer.py             Buffer: NumPy-backed contents, arithmetic/comparison/in-place operators
  texture.py             Texture, Sampler: wrap _mtlpy.Texture/_mtlpy.Sampler
  pipeline.py           Pipeline, CommandBuffer: thin wrappers over _mtlpy.Pipeline/CommandBuffer
  sync.py               Queue, Event, SharedEvent, SharedEventHandle, Fence: advanced synchronization
  operators.py          sqrt/cos/sin/tan/exp/log, sum/max/min/mean reductions
  shader.py             Generates Metal Shading Language source per dtype/texture type
  utils.py              NumPy dtype <-> Metal type/pixel format mapping
tests/               pytest suite
benchmarks/          Standalone performance baseline scripts
examples/            Runnable usage examples
```

Each `Device` in Python owns exactly one `MTL::Device`, one default
`MTL::CommandQueue`, and one `PipelineCache` -- `Device.queue()` creates
additional `MTL::CommandQueue`s on demand for concurrent, independently
scheduled work (see [Advanced synchronization](#advanced-synchronization)).
Buffers default to `MTL::ResourceStorageModeShared`
(`Device.buffer()`/`.empty()`'s `storage=` parameter, `mtlpy.StorageMode`,
picks `MANAGED`/`PRIVATE` instead), so on Apple Silicon's unified memory
there's no copy between CPU and GPU views of the same allocation —
`Buffer.contents` is a NumPy array backed directly by GPU-visible memory
(accessing `.contents` is a true zero-copy, writable view; writing new data
into it via `buf.contents[:] = arr` is still a real memcpy from `arr`'s own
memory, same as it would be for any destination). This zero-copy, writable
view only holds for the default `SHARED` storage — for `PRIVATE`/`MANAGED`
buffers, which have no CPU-visible memory to view directly, `.contents`
transparently materializes a **read-only snapshot** copy instead (writing
into it raises `ValueError` rather than silently doing nothing); see
`Buffer.to_storage()` to convert a `Buffer` between storage modes.

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
- **Zero-copy interop with other GPU libraries**: every `Buffer` implements
  the DLPack protocol, tagged Metal-backed rather than CPU — `mx.asarray(buf,
  copy=False)` (MLX, or any other DLPack-aware library) shares the same
  `id<MTLBuffer>` with no copy. Raw `mtl_ptr` handles are also available for
  non-DLPack native interop — see [Zero-copy interop with other GPU
  libraries](#zero-copy-interop-with-other-gpu-libraries).
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
- **Advanced synchronization**: `Device.queue()` for a second (or third, ...)
  independently-scheduled `MTL::CommandQueue`; `Device.event()`/`.shared_event()`
  to order `CommandBuffer`s against each other -- including across queues,
  and (via `SharedEvent.export_handle()`) across processes; `Device.fence()`
  for same-queue producer/consumer ordering between `Pipeline.run()`
  dispatches -- see [Advanced synchronization](#advanced-synchronization)
  below.
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

## Zero-copy interop with other GPU libraries

Every `Buffer` implements the [DLPack](https://github.com/dmlc/dlpack)
protocol (`__dlpack__`/`__dlpack_device__`), tagged as Metal-backed
(`kDLMetal`) rather than plain CPU memory. Any DLPack-aware library gets a
genuine zero-copy view over the same `id<MTLBuffer>` — no allocation, no
`memcpy` — just by taking the `Buffer` directly. This requires the default
`SHARED` storage mode (see `mtlpy.StorageMode`): a `PRIVATE`/`MANAGED`
`Buffer` raises `BufferError` from `__dlpack__`, since there's no
CPU-visible memory to hand a consumer zero-copy — call
`buf.to_storage(mtlpy.StorageMode.SHARED)` first to get an exportable copy.

```python
import mlx.core as mx

buf = device.buffer(np.arange(8, dtype=np.float32))
arr = mx.asarray(buf, copy=False)  # shares this Buffer's live memory

buf.contents[0] = 111.0
arr[0].item()  # 111.0 -- same underlying allocation, not a copy
```

Measured at 4K/8K image sizes: ~15-25µs regardless of resolution (just
wrapping the existing handle), versus hundreds of µs to a few ms for the
`.contents()`-view-then-copy alternative most non-DLPack-aware code falls
back to — that path scales linearly with buffer size since it's a real
`memcpy`, while the DLPack path doesn't move any bytes at all.

mtlpy itself never imports MLX (or any other consumer) anywhere — the
dunder protocol methods are the complete surface; this is standard DLPack,
not an MLX-specific integration, so any other DLPack-aware library works the
same way.

For interop with something that *isn't* DLPack-aware (e.g. hand-written
PyObjC/Metal bridging code), `Device.mtl_ptr` / `Buffer.mtl_ptr` /
`Texture.mtl_ptr` expose the raw `id<MTLDevice>`/`id<MTLBuffer>`/
`id<MTLTexture>` handle as a plain integer. Unlike the DLPack path (which
retains the underlying Metal object for as long as the consumer holds it),
these are bare pointers with no lifetime management — valid only as long as
you keep the owning mtlpy object referenced yourself.

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

`device.texture(data, pixel_format)`, `.upload()`, and `.download()` are the
simple default path -- always available, and both GPU-side: each stages
through an internal `Buffer` sub-allocated from a `Heap` this `Device` owns
and grows automatically (`Device._staging_buffer()`), then blits, instead of
`MTL::Texture`'s CPU-side `replaceRegion`/`getBytes()` this project used
before measuring the difference (see `benchmarks/blit_bench.py`):

- A *fresh* standalone `Buffer` allocation (`Device.buffer()`/`.empty()`)
  pays a real first-write cost that scales with size -- a brand-new
  `Buffer`'s memory isn't physically committed until the CPU writes it, and
  that first write faults in pages the OS hasn't backed yet. A `Heap`'s
  backing store is committed once, up front, so a *fresh* sub-allocation
  from it skips almost all of that cost.
- Net effect: `.upload()`/`.download()` measured ~2-4x faster than the old
  CPU-side methods at 1080p/4K. They lose only at small sizes (~480p and
  below, where a blit's fixed per-call overhead -- command buffer encode/
  submit/wait -- isn't worth paying yet); `replaceRegion`/`getBytes()`
  aren't exposed as separate methods, since at that point the difference is
  a fraction of a millisecond either way.
- Also means both now work on a `private=True` texture -- `replaceRegion`/
  `getBytes()` can't touch Private storage at all, but a blit doesn't care.
- The internal staging `Buffer` never sticks around: `.upload()`'s is freed
  the instant the (synchronous) blit completes, and `.download()`'s
  contents are copied out into an ordinary, independent array before
  returning (not a zero-copy view -- see its docstring for why: it's what
  lets the internal `Heap` stay small and fixed regardless of how many
  `.download()` results you hold onto, rather than needing capacity for
  all of them). That `Heap` grows to fit the largest single texture you've
  ever processed and never shrinks back down on its own --
  `Device.clear_staging_heap()` reclaims it explicitly if you've done one
  unusually large texture and want that memory back.

There's also a lower-level GPU-side path for explicit control, for when you
need more than what the auto-managed internal `Heap` gives you -- namely,
reusing your *own* `Buffer` across many calls (skips the internal
staging-buffer indirection entirely), or holding several independent
results alive at once (which the internal `Heap` structurally can't do,
since it only ever holds one buffer's worth of space -- see "Maximum
throughput" below):

| Direction | Method | Mechanism | Notes |
|---|---|---|---|
| `Buffer` -> Texture | `tex.upload_from_buffer(buf)` / `device.blit_upload_texture(buf, tex)` | GPU blit | What `.upload()` uses internally, exposed directly so you can supply (and reuse) your own `Buffer` instead of the auto-managed internal one |
| Texture -> `Buffer` | `tex.download_to_buffer(buf)` / `device.blit_download_texture(tex, buf)` | GPU blit | What `.download()` uses internally, exposed directly so you can supply (and reuse) your own `Buffer`. Works on any pixel format (`Unorm` included) and storage mode, including a texture created with `readable=False` (`to_buffer()` below requires `readable=True`) |
| Texture -> `Buffer` | `tex.to_buffer()` | GPU compute kernel | Requires `readable=True` (the copy kernel does `texture.read()`); use this if you want a compute-kernel-based readback specifically, otherwise `download_to_buffer()` above is generally faster (no pipeline dispatch) |
| Texture -> Texture | `src.copy_to(dst)` | GPU blit | Works on any pixel format (including `Unorm`) and any `Shared`/`Private` combination -- e.g. copying a `Shared` texture you populated with `.upload()` into a `Private` one before a hot compute loop |

`empty_texture(..., readable=, writable=, private=)` controls usage flags
and storage mode at creation. `private=True` (`MTLStorageModePrivate`,
GPU-only memory) doesn't restrict `.upload()`/`.download()` or any
`Buffer`-mediated method -- all are blit-based, and a blit doesn't care
about storage mode.

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

## Maximum throughput with several buffers alive at once: `Device.heap()`

`.upload()`/`.download()` already sub-allocate from an internal `Heap`
automatically (see above) — but that internal `Heap` only ever holds *one*
buffer's worth of space at a time, because both methods guarantee their
staging `Buffer` is unreferenced by the time they return. That's exactly
what makes growing it safe without bound-checking against anything still
in use, but it also means it can't help with holding *several* independent
results alive concurrently (e.g. accumulating a batch of
`Texture.download_to_buffer()` results before processing them together,
rather than consuming and discarding each one before the next call) — for
that, build and manage your own `Heap`, sized for your own concurrency:

```python
# Accumulate several downloads before processing them together, without
# paying fresh-standalone-allocation cost for each one.
heap = device.heap(4 * 1024 * 1024, storage=mtlpy.StorageMode.SHARED)

results = []
for tex in textures_to_read_back:
    buf = heap.empty(tex.width * tex.height, np.float32)  # fresh, cheap
    tex.download_to_buffer(buf)
    results.append(buf.numpy())  # each is independent -- safe to hold all of them

process_batch(results)
```

Size the `Heap` for the number of buffers you actually need alive at once —
a `Heap`'s capacity is fixed at creation, and only you know your own
concurrency bound. Guess too small and holding "too many" results alive
raises `RuntimeError: insufficient free space` (confirmed: a `Heap` sized
for 20 buffers fails cleanly, not a crash, on the 21st one held alive
simultaneously) — this is exactly why the internal staging `Heap` behind
`.upload()`/`.download()` doesn't try to solve this case for you; it would
have to guess a limit, and guess wrong for someone. See `Heap`'s class
docstring (`src/mtlpy/heap.py`) for the full measurement and reasoning.

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

## Advanced synchronization

Everything above assumes one `Device`, one queue: `wait=False`/`CommandBuffer`
batching both rely on Metal's guarantee that command buffers on the *same*
`MTL::CommandQueue` retire in commit order. That guarantee doesn't extend
across *different* queues, and sometimes you want more than one -- e.g. two
independent kernels that don't depend on each other, scheduled concurrently
instead of forced through one serial stream. `Device.queue()`, `Device.event()`/
`.shared_event()`, and `Device.fence()` are the tools for that: a second
`MTL::CommandQueue`, and the two Metal primitives for ordering GPU work that
isn't already implied by "same queue, commit order."

### A second queue: `Device.queue()`

```python
q = device.queue()
with device.command_buffer(queue=q) as cb:
    pipeline.run(bufs, grid, cb=cb)
```

Only reachable via the *batched* dispatch path (`Device.command_buffer(queue=q)`
+ `Pipeline.run(..., cb=cb)`) -- a `Pipeline`'s self-contained dispatch
(`pipeline.run(bufs, grid)`, no `cb=`) always targets the `Device`'s own
default queue, regardless of what `Queue`s exist. On its own, a second queue
just gives you a second independently-scheduled submission stream; the
primitives below are what let you order specific pieces of work across it
when you actually need to.

### `Event` / `SharedEvent`: ordering across queues (and processes)

`Device.event()` returns an `MTL::Event` wrapper: a GPU-side-only signal/wait
that `CommandBuffer.signal_event()`/`.wait_for_event()` splice into a batch's
command stream. A `CommandBuffer` on one queue signals an event once its
encoded work completes; a `CommandBuffer` on another queue waits for it
before starting its own -- the one ordering guarantee two independent queues
don't give you for free:

```python
q1, q2 = device.queue(), device.queue()
event = device.event()

with device.command_buffer(queue=q1) as cb1:
    producer.run(bufs, grid, cb=cb1)
    cb1.signal_event(event, 1)

with device.command_buffer(queue=q2) as cb2:
    cb2.wait_for_event(event, 1)   # blocks q2's dispatch until q1's completes
    consumer.run(bufs, grid, cb=cb2)
```

`Device.shared_event()` returns an `MTL::SharedEvent` wrapper -- same
GPU-side `signal_event()`/`wait_for_event()`, plus a CPU-visible `uint64`
"signaled value" you can read/set/block on directly from Python:
`SharedEvent.signal(value)`, `.signaled_value`, and `.wait(value, timeout_ms=5000)`
(blocks the calling thread, releasing the GIL, until `signaled_value` reaches
at least `value`, or the timeout elapses -- returns `False` on timeout). That
makes it the tool for CPU<->GPU handoff in either direction: the GPU signals
something the CPU is waiting on, or the CPU signals something a
`wait_for_event()`-blocked dispatch is waiting on. `SharedEvent` is strictly
more capable than `Event` (superset of what it can do) but costs more to
create/signal -- use `Event` when nothing needs CPU visibility.

`SharedEvent.export_handle()` returns a `SharedEventHandle` another *process*
can import via `Device.import_shared_event(handle)` to synchronize with the
same underlying event -- e.g. coordinating with another framework or a
separate process also using Metal. mtlpy only provides the create/export/
import primitives; actually transporting the handle to the other process
(over an XPC connection, which knows how to encode `MTLSharedEventHandle`
natively since it conforms to `NSSecureCoding`) is up to the caller.

### `Fence`: same-queue producer/consumer ordering

`Device.fence()` returns an `MTL::Fence` wrapper, passed to `Pipeline.run()`
as `wait_fences`/`signal_fences`:

```python
fence = device.fence()
producer.run(bufs, grid, wait=False, signal_fences=[fence])
consumer.run(bufs2, grid, wait=True,  wait_fences=[fence])
```

This orders two dispatches on the *same* `MTL::CommandQueue` (whether
batched into one `CommandBuffer` or each self-contained) without an explicit
`wait=True`/commit-order dependency between them. In practice this is rarely
load-bearing in mtlpy specifically: every `Buffer`/`Texture` here uses
Metal's automatic resource-hazard tracking (none are created with
`hazardTrackingModeUntracked`), which already orders a dispatch that reads a
buffer after an earlier one on the same queue that wrote it, with no fence
needed. `Fence` is exposed as a lower-level, explicit tool for orderings
that aren't implied by resource usage alone. For ordering across
*different* queues, use `Event`/`SharedEvent` instead -- `Fence`'s guarantee
is scoped to a single queue.

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

## License

MIT
