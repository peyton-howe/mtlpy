from __future__ import annotations
import weakref
import numpy as np
from .buffer import Buffer
from .texture import Texture
from . import utils
from .utils import StorageMode, TEXTURE_USAGE_SHADER_READ, TEXTURE_USAGE_SHADER_WRITE


class Heap:
    """A memory pool (MTL::Heap) that Buffers/Textures are sub-allocated
    from instead of each getting its own standalone Metal allocation --
    Metal's own constraint (not one this wrapper adds): every resource
    sub-allocated from the same Heap shares the Heap's storage mode, fixed
    once at Device.heap() time, not chosen per-resource.

    Why this is worth it -- measured, not assumed: a *fresh* standalone
    allocation (Device.buffer()/.empty()) pays a real first-write cost that
    scales with size (e.g. ~1.9ms at 4K just to write into a brand-new
    Buffer's memory the first time), because the CPU write faults in
    physical pages the OS hasn't committed yet. A Heap's backing store is
    already committed once, up front, at Device.heap() time -- so a *fresh*
    sub-allocation from it (a different Buffer/Texture every call, not the
    same one reused) skips almost all of that first-write cost: measured
    within ~4% of reusing a single already-warm Buffer, and ~4x cheaper
    than a fresh standalone allocation, at every size tested. This is
    exactly why Texture.upload()/.download() sub-allocate from an internal,
    auto-growing Heap of their own by default (Device._staging_buffer(),
    see its docstring) instead of a standalone Buffer -- you get this win
    automatically without ever touching this class directly.

    Why you'd still want your *own* Heap on top of that automatic one: the
    internal staging Heap behind .upload()/.download() only ever holds
    *one* buffer's worth of space at a time (both methods guarantee their
    staging Buffer is unreferenced before they return) -- it structurally
    can't help if you want several independent results alive at once (e.g.
    accumulating .download_to_buffer() results in a list before processing
    them together). A Heap has a *fixed* capacity chosen at creation time,
    so a shared internal Heap sized for "however many mtlpy guessed" would
    eventually run out and raise "insufficient free space" under a pattern
    that works fine today with standalone allocation (confirmed: a Heap
    sized for 20 buffers fails on the 21st one held alive simultaneously,
    with a clean RuntimeError, not a crash -- but a failure all the same).
    The right heap size is inherently workload-dependent (how many buffers
    do *you* need alive at once?), which only the caller can answer -- so
    mtlpy doesn't guess it for you inside a stateless method. If your
    workload needs several independent results alive concurrently and you
    know your own concurrency bound, size a Heap for it yourself and reuse
    it across many upload_from_buffer()/download_to_buffer() calls; if you
    only need one buffer at a time (the common case), .upload()/.download()
    already give you this Heap's performance for free -- no manual Heap
    required.

    This is intentionally the minimal Metal heap surface: automatic-type
    heaps only (no placement/sparse heaps), no aliasing control (every
    resource is non-aliased, Metal's default), no purgeability API."""

    def __init__(self, _heap, storage: StorageMode, device, *, _weak_device: bool = False):
        self._heap   = _heap    # _mtlpy.Heap
        self.storage = StorageMode(storage)
        # _weak_device is for Device._staging_buffer()'s internal use only
        # (see its own call site) -- it caches its lazily-grown Heap as
        # Device._staging_heap, and a strong Heap -> Device reference here
        # would close a cycle (Device -> _staging_heap -> Heap -> Device)
        # that pure refcounting can never break on its own, same issue and
        # same fix as Pipeline._device_ref in pipeline.py (see its comment
        # for the full story). That internal Heap is never handed to a
        # caller, so a weakref is safe there: every access to it happens
        # from inside a Device method, where Device is definitionally alive.
        #
        # A Heap from the public Device.heap(), by contrast, must hold its
        # Device *strongly* -- Device doesn't cache those back (no cycle
        # risk), and every other resource in this library (Buffer, Texture,
        # ...) already keeps its owning Device alive for as long as it's
        # held; a weakref here would silently break that for any caller who
        # holds a Heap without also separately holding its Device (e.g. a
        # factory function that returns just the Heap) -- confirmed by
        # testing: the Device wrapper was freed out from under the Heap,
        # and self._device silently became None instead of a clear error.
        self._device_ref = weakref.ref(device) if _weak_device else (lambda: device)

    @property
    def _device(self):
        return self._device_ref()

    @property
    def size(self) -> int:
        """Actual heap size in bytes as allocated by Metal -- may be larger
        than the size_bytes passed to Device.heap() (Metal rounds up to its
        own alignment/page granularity)."""
        return self._heap.size

    @property
    def used_size(self) -> int:
        """Bytes currently sub-allocated to live Buffers/Textures from this
        Heap -- drops back down as soon as a Buffer/Texture from this Heap
        is garbage collected (CPython refcounting makes this deterministic:
        it happens the instant the last reference to it goes away, not at
        some later GC pass), same as any other Buffer/Texture's underlying
        Metal allocation freeing on collection. A Buffer/Texture you don't
        keep a reference to won't show up here for long."""
        return self._heap.used_size

    def buffer(self, data: np.ndarray | int, dtype=None) -> Buffer:
        """Same as Device.buffer(), but sub-allocated from this Heap -- no
        storage= param: the result's storage is always this Heap's storage
        (see the class docstring)."""
        if isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data)
            buf = self.empty(arr.shape, arr.dtype)
            if self.storage == StorageMode.SHARED:
                buf.contents[:] = arr.reshape(-1)  # buf.contents is always flat
                return buf
            # Same staging-then-blit pattern Device.buffer() uses for
            # non-Shared storage (see its docstring), except the staging
            # Buffer is a plain standalone allocation (Device.buffer(),
            # not this Heap) -- only the final result needs to live on
            # the heap.
            staging = self._device.buffer(arr)
            self._device.copy_buffer(staging, buf)
            return buf
        size = int(data)
        dt   = utils.to_numpy(dtype)
        raw  = self._heap.new_buffer(size * np.dtype(dt).itemsize)
        return Buffer(raw, dt, (size,), self._device)

    def empty(self, size: int | tuple[int, ...], dtype) -> Buffer:
        shape = (int(size),) if isinstance(size, (int, np.integer)) else tuple(int(s) for s in size)
        flat_size = utils.shape_size(shape)
        dt  = utils.to_numpy(dtype)
        raw = self._heap.new_buffer(flat_size * np.dtype(dt).itemsize)
        return Buffer(raw, dt, shape, self._device)

    def empty_texture(self, shape: tuple[int, ...], pixel_format: str, *,
                       readable: bool = True, writable: bool = True) -> Texture:
        """Same as Device.empty_texture(), but sub-allocated from this Heap
        -- no private= param: this Heap's storage mode applies to every
        resource on it, same as Heap.buffer()."""
        dims = len(shape)
        if dims not in (1, 2, 3):
            raise ValueError(
                f"Texture shape must have 1, 2, or 3 dims (spatial only -- "
                f"exclude the channel axis, which pixel_format implies), got {shape}"
            )
        if not readable and not writable:
            raise ValueError("A texture must be at least one of readable/writable")
        info = utils.pixel_format_info(pixel_format)
        width  = shape[-1]
        height = shape[-2] if dims >= 2 else 1
        depth  = shape[-3] if dims >= 3 else 1
        usage = (TEXTURE_USAGE_SHADER_READ if readable else 0) | \
                (TEXTURE_USAGE_SHADER_WRITE if writable else 0)
        raw = self._heap.new_texture(dims, info.mtl_value, width, height, depth, usage)
        return Texture(raw, dims, pixel_format, width, height, depth, self._device,
                        readable=readable, writable=writable)

    def __repr__(self) -> str:
        return (f"Heap(size={self.size}, used_size={self.used_size}, "
                f"storage={self.storage.name.lower()})")
