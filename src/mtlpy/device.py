from __future__ import annotations
import numpy as np
from .binary_archive import BinaryArchive
from .buffer import Buffer
from .capture import Capture, CaptureScope
from .heap import Heap
from .pipeline import CommandBuffer, Pipeline
from .sync import Event, Fence, Queue, SharedEvent, SharedEventHandle
from .texture import Sampler, Texture
from . import utils, shader
from .utils import StorageMode, TEXTURE_USAGE_SHADER_READ, TEXTURE_USAGE_SHADER_WRITE

try:
    from . import _mtlpy
except ImportError as e:
    raise ImportError(
        "mtlpy C extension not found. Install with: pip install mtlpy"
    ) from e


def _texture_buffer_layout(tex: Texture, buf: Buffer, offset: int) -> tuple[int, int]:
    """Shared validation for Device.blit_upload_texture()/blit_download_texture()
    (and Texture.upload_from_buffer(), which delegates to the former): buf
    must hold tex's data tightly packed (dtype matching tex's per-channel
    dtype) starting at a byte offset into buf. Returns the
    bytes_per_row/bytes_per_image describing that layout, for the blit
    encoder call."""
    if buf.dtype != tex.dtype:
        raise TypeError(
            f"Buffer dtype {buf.dtype} doesn't match texture pixel_format "
            f"{tex.pixel_format!r}'s per-channel dtype {tex.dtype}"
        )
    expected_bytes = utils.shape_size(tex.shape) * tex.dtype.itemsize
    buf_bytes = buf.size * buf.dtype.itemsize
    if offset + expected_bytes > buf_bytes:
        raise ValueError(
            f"Buffer has {buf_bytes} bytes, but offset={offset} plus this "
            f"{tex.shape} texture's {expected_bytes} bytes needs "
            f"{offset + expected_bytes}"
        )
    return tex._bytes_per_row_and_image()


def list_devices() -> list[str]:
    """Names of all Metal-capable GPUs on this machine, in the order
    Device(index=...) expects. On most Macs (a single integrated GPU) this
    returns exactly one name; multi-GPU Macs (e.g. with an eGPU) list more."""
    return _mtlpy.list_devices()


class Device:
    def __init__(self, index: int | None = None, cache_path: str | bool | None = None):
        """index selects a specific GPU from list_devices() (for multi-GPU
        machines); the default (None) uses the system default GPU.

        cache_path controls where this Device's on-disk compiled-pipeline
        cache lives (see flush_cache()/.pipeline_cache_path): the default
        (None) uses ~/Library/Caches/mtlpy/pipelines.metallib; False
        disables on-disk caching entirely (compiled pipelines are still
        deduped in memory for this process, just never written to/read from
        disk); any other value is used as a custom path instead."""
        if cache_path is False:
            resolved_cache_path: str | None = ""
        elif cache_path is None:
            resolved_cache_path = None
        elif cache_path is True:
            raise TypeError(
                "cache_path=True is not a valid value -- pass a path string for a "
                "custom location, False to disable on-disk caching, or omit it "
                "(or pass None) for the default location"
            )
        else:
            resolved_cache_path = str(cache_path)
        self._dev = _mtlpy.Device(-1 if index is None else index, resolved_cache_path)
        # Compiled texture_to_buffer_kernel Pipelines, keyed by the (dims,
        # read_t, store_t, channels, normalized) signature buffer_from_texture()
        # generates MSL source from -- avoids re-generating an identical
        # source string and re-hitting the C++ PipelineCache's string-hash
        # lookup on every to_buffer() call for the same texture shape/format
        # (e.g. repeated per-frame GPU readback).
        self._texture_to_buffer_pipelines: dict = {}
        # Backs _staging_buffer() -- see its docstring. Grown lazily, never
        # shrunk: bounded by the largest single Texture.upload()/.download()
        # this Device has ever done, not by how many have happened.
        self._staging_heap: Heap | None = None

    @property
    def mtl_ptr(self) -> int:
        """The id<MTLDevice> handle itself, as a raw integer. Any
        Buffer/Texture created by this Device belongs to this same MTLDevice
        -- external native code consuming one of them directly (see
        Texture.mtl_ptr) needs this to, e.g., set a CAMetalLayer's .device
        to match, or otherwise confirm it's working with the right physical
        GPU (Metal forbids mixing resources from different devices in one
        command encoder). Same raw-pointer, no-automatic-lifetime caveat as
        Texture.mtl_ptr: valid only as long as this Device is kept alive."""
        return self._dev.mtl_ptr

    def __enter__(self) -> Device:
        return self

    def __exit__(self, *exc_info) -> None:
        self.flush_cache()

    def flush_cache(self, path: str | None = None) -> None:
        """Serialize the on-disk compiled-pipeline cache now, rather than
        waiting for this Device to be garbage collected. Useful for a
        long-running process that wants newly-compiled kernels to survive
        a crash, or just deterministic cleanup via `with mtlpy.Device() as d:`.

        path (default None) overrides the destination for this call only,
        without changing where future flush_cache() calls (or garbage
        collection) write to -- see Device(cache_path=...) to change that
        permanently."""
        self._dev.flush_cache(path)

    @property
    def pipeline_cache_size(self) -> int:
        """Number of distinct (source, function_name) pipelines currently
        cached in memory for this Device -- includes ones the on-disk
        archive hasn't necessarily been given a chance to persist yet (see
        flush_cache())."""
        return self._dev.pipeline_cache_size

    @property
    def pipeline_cache_path(self) -> str:
        """Where this Device's on-disk compiled-pipeline cache lives --
        empty string if disabled (Device(cache_path=False)) or
        undeterminable (e.g. $HOME unset)."""
        return self._dev.pipeline_cache_path

    def binary_archive(self, path: str | None = None) -> BinaryArchive:
        """A user-managed MTL::BinaryArchive independent of this Device's
        own internal pipeline cache -- see BinaryArchive's class docstring.
        path to an existing file opens it; omitted (or a path that doesn't
        exist yet) starts a fresh, empty archive."""
        return BinaryArchive(self._dev.create_binary_archive(path), self)

    def buffer(self, data: np.ndarray | int, dtype=None,
               storage: StorageMode = StorageMode.SHARED) -> Buffer:
        """storage -- see mtlpy.StorageMode. When data is an ndarray and
        storage isn't SHARED, the array is first staged into a Shared buffer
        (the only storage mode a plain CPU memcpy can write into) and then
        GPU-blit-copied into a fresh buffer of the requested storage (see
        Buffer.to_storage())."""
        storage = StorageMode(storage)
        if isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data)
            buf = self.empty(arr.shape, arr.dtype)  # always Shared -- see docstring above
            buf.contents[:] = arr.reshape(-1)  # buf.contents is always flat
            return buf.to_storage(storage)
        size = int(data)
        dt   = utils.to_numpy(dtype)
        raw  = self._dev.create_buffer(size * np.dtype(dt).itemsize, int(storage))
        return Buffer(raw, dt, (size,), self)

    def empty(self, size: int | tuple[int, ...], dtype,
              storage: StorageMode = StorageMode.SHARED) -> Buffer:
        """storage -- see mtlpy.StorageMode."""
        shape = (int(size),) if isinstance(size, (int, np.integer)) else tuple(int(s) for s in size)
        flat_size = utils.shape_size(shape)
        dt  = utils.to_numpy(dtype)
        raw = self._dev.create_buffer(flat_size * np.dtype(dt).itemsize, int(StorageMode(storage)))
        return Buffer(raw, dt, shape, self)

    def _staging_buffer(self, n_elements: int, dtype) -> Buffer:
        """Internal Shared Buffer, sub-allocated from a lazily-grown Heap
        this Device owns -- backs Texture.upload()/.download()'s default
        implementations (see their docstrings for the measured win over a
        standalone Device.buffer()/.empty() allocation: a Heap's backing
        store is committed once, up front, so a fresh sub-allocation from
        it skips almost all of the first-write page-fault cost a brand-new
        standalone allocation pays).

        Only ever holds ONE buffer's worth of space at a time: both
        callers guarantee the returned Buffer is unreferenced by the time
        they return (upload() blits synchronously then lets it go;
        download() copies its contents out before returning, see its
        docstring) -- so growing this heap to fit a bigger request is
        always safe, no previously-handed-out sub-allocation can still be
        alive when that happens. This is NOT the tool for holding several
        independent results alive at once -- see Device.heap()'s own
        docstring for that case, which this deliberately doesn't attempt
        to solve (this heap's size tracks the single largest request ever
        made, not how many were made).

        Grows to fit the largest request seen so far and never shrinks on
        its own -- see clear_staging_heap() to reclaim that memory
        explicitly (e.g. after one unusually large texture, before doing
        many more small ones)."""
        nbytes = n_elements * np.dtype(dtype).itemsize
        if self._staging_heap is None or self._staging_heap.size < nbytes:
            self._staging_heap = self.heap(nbytes, storage=StorageMode.SHARED)
        return self._staging_heap.empty(n_elements, dtype)

    def clear_staging_heap(self) -> int:
        """Frees _staging_buffer()'s internal Heap (used by
        Texture.upload()/.download(), see its docstring) right now,
        instead of it sitting at its high-water-mark size for the rest of
        this Device's lifetime -- _staging_buffer() grows that Heap to fit
        the largest request ever made and never shrinks it back down on
        its own. The next .upload()/.download() call after this lazily
        recreates it, sized for whatever it needs at that point -- so this
        only matters if you've done one unusually large texture and want
        that memory back before doing many more small ones. Returns the
        number of bytes freed (0 if there was no staging Heap yet, i.e.
        .upload()/.download() were never called, or this was already
        called since the last one that was)."""
        freed = self._staging_heap.size if self._staging_heap is not None else 0
        self._staging_heap = None
        return freed

    def copy_buffer(self, src: Buffer, dst: Buffer, *, src_offset: int = 0, dst_offset: int = 0,
                     size_bytes: int | None = None, wait: bool = True) -> None:
        """Hardware-blit buffer-to-buffer copy (MTLBlitCommandEncoder::
        copyFromBuffer) -- works for any combination of storage modes on
        either side, including Private (which has no CPU-visible memory a
        plain memcpy could reach). This is the mechanism Buffer.to_storage()
        uses internally to materialize a Shared copy of a Private/Managed
        Buffer; exposed here directly for any other buffer-to-buffer copy,
        e.g. landing one Buffer's data at a specific offset into a larger
        one via src_offset/dst_offset (mirroring Texture.upload_from_buffer()'s
        offset param).

        size_bytes defaults to everything from src_offset to the end of src
        (src.size * src.dtype.itemsize - src_offset). dst must have room
        for dst_offset + size_bytes -- out of range raises RuntimeError
        (validated on the C++ side, see csrc/device.cpp) rather than
        corrupting memory or crashing."""
        if src._device is not dst._device:
            raise ValueError(
                "Buffers belong to different Device instances -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        if size_bytes is None:
            size_bytes = src.size * src.dtype.itemsize - src_offset
        self._dev.copy_buffer(src._buf, src_offset, dst._buf, dst_offset, size_bytes, wait)

    def compile(self, source: str, function_name: str, archive: BinaryArchive | None = None) -> Pipeline:
        """archive (default None), if given, additionally registers the
        compiled pipeline into that BinaryArchive -- on top of this
        Device's own internal cache, which always happens regardless. See
        BinaryArchive's class docstring for why you'd want a second,
        explicit archive."""
        raw = self._dev.compile(source, function_name, archive._archive if archive is not None else None)
        return Pipeline(raw, self)

    def heap(self, size_bytes: int, storage: StorageMode = StorageMode.SHARED) -> Heap:
        """A memory pool (MTL::Heap) of at least size_bytes to sub-allocate
        Buffers/Textures from via Heap.buffer()/.empty()/.empty_texture()
        -- see Heap's class docstring for when this is worth it over the
        standalone allocations Device.buffer()/.empty()/.empty_texture()
        make. storage applies to every resource sub-allocated from the
        returned Heap (Metal's own constraint, not chosen per-resource);
        PRIVATE is the most common choice for a heap in practice (GPU-only
        scratch memory reused across many allocate/free cycles), but this
        defaults to SHARED for consistency with Device.buffer()/.empty()'s
        own default -- pass storage=StorageMode.PRIVATE explicitly for the
        common case."""
        raw = self._dev.create_heap(size_bytes, int(StorageMode(storage)))
        return Heap(raw, storage, self)

    def empty_texture(self, shape: tuple[int, ...], pixel_format: str, *,
                       readable: bool = True, writable: bool = True,
                       private: bool = False) -> Texture:
        """shape is the *spatial* shape only -- (width,), (height, width),
        or (depth, height, width) for a 1D/2D/3D texture; channel count
        comes from pixel_format (e.g. "rgba8Unorm" is 4-channel) and isn't
        part of shape. len(shape) determines dims.

        readable/writable declare which of access::read/access::write a
        kernel will actually use on this texture -- default both True
        (matches every prior version of this method), but narrowing to just
        what's needed lets Metal keep a more aggressive internal
        tiled/compressed layout for the texture, since it no longer has to
        stay generically read+write-capable (measured ~2x faster GPU-side
        for a read-only 9-tap stencil kernel vs the same texture declared
        read+write). private=True additionally uses MTL::StorageModePrivate
        (GPU-only memory, freeing Metal to optimize further) instead of the
        default StorageModeShared -- doesn't restrict .upload()/.download()
        at all (both are blit-based, not replaceRegion/getBytes-based --
        see their own docstrings), so a private texture works with every
        method on this class the same as any other."""
        dims = len(shape)
        if dims not in (1, 2, 3):
            raise ValueError(
                f"Texture shape must have 1, 2, or 3 dims (spatial only -- "
                f"exclude the channel axis, which pixel_format implies), got {shape}"
            )
        if not readable and not writable:
            raise ValueError("A texture must be at least one of readable/writable")
        info = utils.pixel_format_info(pixel_format)
        # Note: this deliberately does NOT try to detect "shape still has a
        # trailing channel axis" (e.g. a 2D (H, W, C) array's full shape
        # passed by mistake) for dims==3 -- a prior version compared
        # shape[-1] to info.channels, which also incorrectly rejected any
        # *genuine* 3D texture whose width happened to equal the format's
        # channel count (e.g. (10, 20, 4) for "rgba8Unorm", a real
        # depth=10/height=20/width=4 volume texture). The two cases are not
        # distinguishable from shape alone, and rejecting valid input is
        # worse than missing a hint for a mistake -- use
        # device.texture(data, pixel_format) for the "still has a channel
        # axis" case instead, which strips it unambiguously.
        width  = shape[-1]
        height = shape[-2] if dims >= 2 else 1
        depth  = shape[-3] if dims >= 3 else 1
        usage = (TEXTURE_USAGE_SHADER_READ if readable else 0) | \
                (TEXTURE_USAGE_SHADER_WRITE if writable else 0)
        raw = self._dev.create_texture(dims, info.mtl_value, width, height, depth,
                                        usage, private)
        return Texture(raw, dims, pixel_format, width, height, depth, self,
                        readable=readable, writable=writable)

    def texture(self, data: np.ndarray, pixel_format: str) -> Texture:
        """Create a texture matching data's shape and upload it in one call.
        data's last axis is the channel axis if pixel_format is multi-channel
        (e.g. an (H, W, 4) array for "rgba8Unorm"), matching Texture.shape."""
        info = utils.pixel_format_info(pixel_format)
        spatial_shape = data.shape[:-1] if info.channels > 1 else data.shape
        tex = self.empty_texture(spatial_shape, pixel_format)
        tex.upload(data)
        return tex

    def blit_upload_texture(self, buf: Buffer, tex: Texture, *, offset: int = 0, wait: bool = True) -> None:
        """Hardware-blit upload: copies buf's data into tex via
        MTLBlitCommandEncoder, instead of Texture.upload()'s CPU-side
        replaceRegion copy. This is the mechanism Texture.upload_from_buffer()
        uses internally (see that method's docstring for the buf-layout/
        offset contract, which this validates identically); exposed here
        directly for callers who'd rather reach it as a Device method."""
        if buf._device is not tex._device:
            raise ValueError(
                "Buffer and Texture belong to different Device instances -- Metal "
                "does not allow referencing resources from different MTLDevice "
                "objects in the same command buffer"
            )
        bytes_per_row, bytes_per_image = _texture_buffer_layout(tex, buf, offset)
        self._dev.blit_upload_texture(buf._buf, offset, tex._tex, bytes_per_row, bytes_per_image, wait)

    def blit_download_texture(self, tex: Texture, buf: Buffer, *, offset: int = 0, wait: bool = True) -> None:
        """The read counterpart to blit_upload_texture(): hardware-blit copy
        of tex's pixel data into buf, instead of Texture.download()'s
        CPU-side getBytes() copy or buffer_from_texture()/.to_buffer()'s
        compute-kernel readback. Three concrete advantages over the latter:
        no compute-pipeline dispatch at all (pure copy-engine transfer);
        works on a tex created with readable=False (buffer_from_texture()
        requires MTLTextureUsageShaderRead and raises otherwise -- a blit
        copy needs no shader access at all); and lands directly into an
        existing buf you already own, at any offset, instead of always
        allocating a fresh Buffer. Works for any pixel format (Unorm
        included) and any combination of Shared/Private storage on either
        side, same as copy_texture(). buf must have room for this texture's
        tightly packed bytes starting at offset (same layout
        Texture.upload_from_buffer() expects, same validation) -- see
        Texture.download_to_buffer()."""
        if tex._device is not buf._device:
            raise ValueError(
                "Texture and Buffer belong to different Device instances -- Metal "
                "does not allow referencing resources from different MTLDevice "
                "objects in the same command buffer"
            )
        bytes_per_row, bytes_per_image = _texture_buffer_layout(tex, buf, offset)
        self._dev.blit_download_texture(tex._tex, buf._buf, offset, bytes_per_row, bytes_per_image, wait)

    def buffer_from_texture(self, tex: Texture) -> Buffer:
        """GPU-side texture readback: dispatches a small compute kernel that
        copies tex's pixels into a tightly packed Buffer, instead of the
        CPU-side getBytes() copy tex.download()/.numpy() do. tex keeps
        whatever internal layout Metal chose for it (may be tiled/swizzled
        for texture-cache locality) -- this only touches memory on the GPU
        side, landing the result in a genuinely zero-copy Buffer.contents/
        .numpy() with no further CPU copy needed. See shader.
        texture_to_buffer_kernel for why this needs both tex's MSL read type
        and its narrower storage type.

        Requires tex.readable (raises otherwise): the copy kernel does
        texture.read(), which needs MTLTextureUsageShaderRead -- a texture
        created via Device.empty_texture(readable=False, ...) doesn't have
        it, and Metal would otherwise fail the dispatch with an opaque
        validation error instead of this clear message."""
        if tex._device is not self:
            raise ValueError(
                "Texture belongs to a different Device instance -- Metal does not "
                "allow referencing resources from different MTLDevice objects in "
                "the same command buffer"
            )
        if not tex.readable:
            raise ValueError(
                f"Texture was created with readable=False (see "
                f"Device.empty_texture()), so it lacks MTLTextureUsageShaderRead -- "
                f"buffer_from_texture()'s copy kernel needs to read it. Create the "
                f"texture with readable=True (the default) to read it back this way."
            )
        store_t   = utils.msl_storage_type(tex.dtype)
        cache_key = (tex.dims, tex.msl_scalar_type, store_t, tex.channels, tex.normalized)
        pipeline  = self._texture_to_buffer_pipelines.get(cache_key)
        if pipeline is None:
            source   = shader.texture_to_buffer_kernel(
                tex.dims, tex.msl_scalar_type, store_t, tex.channels, tex.normalized)
            pipeline = self.compile(source, "texture_to_buffer")
            self._texture_to_buffer_pipelines[cache_key] = pipeline
        out  = self.empty(tex.shape, tex.dtype)
        grid = [tex.width,
                tex.height if tex.dims >= 2 else 1,
                tex.depth  if tex.dims >= 3 else 1]
        pipeline.run([out], grid, textures=[tex])
        return out

    def sampler(self, linear: bool = True, repeat: bool = False) -> Sampler:
        """linear=False uses nearest-neighbor filtering; repeat=True wraps
        out-of-bounds texture coordinates instead of clamping to the edge."""
        raw = self._dev.create_sampler(linear, repeat)
        return Sampler(raw, linear, repeat, self)

    def command_buffer(self, queue: Queue | None = None) -> CommandBuffer:
        """A batch of Pipeline.run() dispatches that share one MTLCommandBuffer
        submission -- see CommandBuffer's docstring for the context-manager
        usage. Use this instead of separate Pipeline.run() calls when you
        have multiple dispatches that always run together (e.g. a multi-pass
        kernel), to pay one command-buffer-submit + wait instead of one per
        dispatch.

        queue (default None) submits this batch on a secondary Queue (see
        Device.queue()) instead of this Device's own default queue -- the
        mechanism for running independent streams of work on more than one
        MTL::CommandQueue, synchronized only where a CommandBuffer.wait_for_event()/
        .signal_event() call says so."""
        if queue is not None and queue._device is not self:
            raise ValueError(
                "Queue belongs to a different Device instance -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        raw = self._dev.create_command_buffer(queue._queue if queue is not None else None)
        return CommandBuffer(raw, self)

    def queue(self) -> Queue:
        """A second MTL::CommandQueue beyond this Device's own default one --
        see Queue's class docstring for what it's for and its one
        restriction (only reachable via the batched Device.command_buffer(queue=...)
        path, not a Pipeline's self-contained dispatch)."""
        return Queue(self._dev.create_queue(), self)

    def event(self) -> Event:
        """A GPU-side-only synchronization primitive for ordering work
        across separate CommandBuffers/Queues -- see Event's class
        docstring for the producer/consumer pattern. Cheaper than
        shared_event() when nothing needs to read/wait on it from the CPU."""
        return Event(self._dev.create_event(), self)

    def shared_event(self) -> SharedEvent:
        """Like event(), but adds CPU-visible signal()/.signaled_value/
        wait() for CPU<->GPU handoff, and export_handle() for cross-process
        use -- see SharedEvent's class docstring."""
        return SharedEvent(self._dev.create_shared_event(), self)

    def import_shared_event(self, handle: SharedEventHandle) -> SharedEvent:
        """The receiving end of SharedEvent.export_handle(): reconstructs
        the same underlying MTL::SharedEvent from a handle another process
        exported (see SharedEventHandle's docstring for how the handle
        itself needs to reach this process)."""
        return SharedEvent(self._dev.create_shared_event_from_handle(handle._handle), self)

    def fence(self) -> Fence:
        """A same-queue producer/consumer ordering primitive for
        Pipeline.run's wait_fences/signal_fences -- see Fence's class
        docstring for what it does and doesn't guarantee."""
        return Fence(self._dev.create_fence(), self)

    def start_capture(self, path: str | None = None) -> Capture:
        """Starts a GPU frame capture (MTLCaptureManager) covering every
        dispatch/blit on this Device from now until stop_capture(). path
        (default None) captures to a .gputrace file at that location,
        openable later in Xcode; omitted, captures live to an attached
        Xcode debugger's GPU debugger instead (raises if none is attached).
        Either way requires the MTL_CAPTURE_ENABLED=1 environment variable
        to be set for this process -- Metal disables programmatic capture
        entirely otherwise, regardless of destination.

        Returns a Capture usable as an optional context manager for the
        matching stop_capture() call:

            with device.start_capture("trace.gputrace"):
                pipeline.run(bufs, grid)
            # stop_capture() already called here

        Calling device.stop_capture() directly (ignoring the return value)
        works exactly the same."""
        self._dev.start_capture(path)
        return Capture(self)

    def stop_capture(self) -> None:
        """Ends a capture started by start_capture() -- on *any* Device:
        this is Metal's own process-wide MTLCaptureManager, not per-Device
        state, so it stops whatever capture is currently active regardless
        of which Device.start_capture() call began it."""
        self._dev.stop_capture()

    @property
    def is_capturing(self) -> bool:
        """Whether a GPU capture is currently active anywhere in this
        process -- not scoped to this Device specifically, same caveat as
        stop_capture()."""
        return self._dev.is_capturing()

    def capture_scope(self, label: str | None = None, queue: Queue | None = None) -> CaptureScope:
        """A labeled begin/end marker for Xcode's GPU debugger timeline --
        see CaptureScope's class docstring. queue (default None) scopes it
        to a secondary Queue (see Device.queue()) instead of this Device's
        own default queue/command-buffer activity."""
        if queue is not None and queue._device is not self:
            raise ValueError(
                "Queue belongs to a different Device instance -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        raw = self._dev.create_capture_scope(label, queue._queue if queue is not None else None)
        return CaptureScope(raw, self)

    def _binary_op(self, name: str, shader_fn, a: Buffer, b: Buffer, out: Buffer | None = None) -> Buffer:
        if a._device is not b._device:
            raise ValueError(
                "Buffers belong to different Device instances -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        # Checked by .size, not .shape: a Metal buffer has no shape of its
        # own (it's just bytes) -- .shape is Python-side metadata, and two
        # equal-size buffers with different declared shapes are still
        # perfectly valid operands.
        if a.size != b.size:
            raise ValueError(f"Buffer size mismatch: {a.size} != {b.size}")
        if a.dtype != b.dtype:
            raise TypeError(f"Buffer dtype mismatch: {a.dtype} != {b.dtype}")
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        if out is None:
            # Matches a's storage, not the default -- an op on Private
            # operands should produce a Private result, not silently
            # downgrade to Shared (see Buffer.to_storage()/StorageMode).
            out = self.empty(a.shape, a.dtype, storage=a.storage)
        # Safe to alias out with a/b: each GPU thread reads then writes only
        # its own index, so an in-place dispatch (out is a or b) has no
        # cross-thread data hazard.
        pipeline.run([a, b, out], a.size)
        return out

    def _scalar_op(self, name: str, shader_fn, a: Buffer, scalar, out: Buffer | None = None) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        # Always Shared regardless of a's storage: this needs a plain CPU
        # write (.contents[:] = ...) to stage the scalar value, which only
        # Shared storage supports -- see Device.buffer()'s ndarray path.
        # Mixing a Shared operand with a non-Shared a/out in one dispatch is
        # ordinary, fully-supported Metal usage (hazard tracking is
        # per-resource, not storage-mode-dependent).
        scalar_buf = self.buffer(np.array([scalar], dtype=a.dtype))
        if out is None:
            out = self.empty(a.shape, a.dtype, storage=a.storage)
        pipeline.run([a, scalar_buf, out], a.size)
        return out

    def _negate_op(self, a: Buffer) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader.negate_kernel(metal_type), "negate")
        out        = self.empty(a.shape, a.dtype, storage=a.storage)
        pipeline.run([a, out], a.size)
        return out

    def _compare_op(self, name: str, shader_fn, a: Buffer, b: Buffer) -> Buffer:
        if a._device is not b._device:
            raise ValueError(
                "Buffers belong to different Device instances -- Metal does not "
                "allow sharing resources across MTLDevice objects"
            )
        if a.size != b.size:
            raise ValueError(f"Buffer size mismatch: {a.size} != {b.size}")
        if a.dtype != b.dtype:
            raise TypeError(f"Buffer dtype mismatch: {a.dtype} != {b.dtype}")
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        out        = self.empty(a.shape, np.bool_, storage=a.storage)
        pipeline.run([a, b, out], a.size)
        return out

    def _compare_scalar_op(self, name: str, shader_fn, a: Buffer, scalar) -> Buffer:
        metal_type = utils.to_metal(a.dtype)
        pipeline   = self.compile(shader_fn(metal_type), name)
        # See _scalar_op's comment -- scalar_buf must stay Shared regardless
        # of a's storage.
        scalar_buf = self.buffer(np.array([scalar], dtype=a.dtype))
        out        = self.empty(a.shape, np.bool_, storage=a.storage)
        pipeline.run([a, scalar_buf, out], a.size)
        return out

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._dev.max_threads_per_threadgroup()
