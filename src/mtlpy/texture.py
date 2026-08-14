from __future__ import annotations
import numpy as np
from . import utils


class Texture:
    """A Metal texture (1D/2D/3D). Unlike Buffer, a Texture's CPU-visible
    memory layout isn't guaranteed to be a tightly packed array (Metal may
    pad/tile rows internally), so there's no Buffer.contents equivalent --
    .upload()/.download() are both hardware blits through an internal
    staging Buffer (see their own docstrings), not a live view over GPU
    memory, and not the CPU-side replaceRegion/getBytes this project used
    before measuring that a Heap-staged blit beats both. Works regardless
    of storage mode, including private=True."""

    def __init__(self, _tex, dims: int, pixel_format: str,
                 width: int, height: int, depth: int, device,
                 readable: bool = True, writable: bool = True):
        self._tex         = _tex          # _mtlpy.Texture
        self.dims         = dims
        self.pixel_format = pixel_format
        self.width        = width
        self.height       = height
        self.depth        = depth
        info = utils.pixel_format_info(pixel_format)
        self.channels         = info.channels
        self.dtype            = info.dtype
        self.normalized       = info.normalized
        self.msl_scalar_type  = info.msl_scalar_type
        self.is_private       = _tex.is_private
        # What Device.empty_texture()'s readable/writable declared this
        # texture's MTLTextureUsage as -- checked by buffer_from_texture()
        # before generating a kernel that does texture.read(), since a
        # texture created with readable=False lacks MTLTextureUsageShaderRead
        # and Metal rejects the dispatch (previously an opaque GPU-side
        # validation failure instead of a clear Python exception).
        self.readable          = readable
        self.writable          = writable
        self._device          = device

    @property
    def mtl_ptr(self) -> int:
        """The id<MTLTexture> handle itself, as a raw integer -- for handing
        this Texture to code outside mtlpy (e.g. a hand-written PyObjC/Metal
        bridge embedding a CAMetalLayer in a Qt/GTK/etc. window) that wants
        to do its own native interop, without mtlpy needing to know anything
        about that consumer.

        Unlike Buffer.__dlpack__, there's no automatic lifetime management
        attached to this pointer -- it's valid only as long as this Texture
        object is kept alive Python-side (nothing retains the underlying
        id<MTLTexture> on your behalf). If the consumer needs to outlive
        this Texture, it must take its own retain (e.g. PyObjC's
        objc.objc_object(c_void_p=...) bridging does this automatically when
        it wraps the pointer; raw ctypes access does not).

        The texture also belongs to a specific MTLDevice -- see
        Device.mtl_ptr -- Metal forbids referencing it from any other
        device's command encoder."""
        return self._tex.mtl_ptr

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial dims (numpy image convention: (H, W) row-major, depth
        first for 3D) plus a trailing channel dim if channels > 1."""
        spatial = {
            1: (self.width,),
            2: (self.height, self.width),
            3: (self.depth, self.height, self.width),
        }[self.dims]
        return spatial + (self.channels,) if self.channels > 1 else spatial

    def _bytes_per_row_and_image(self) -> tuple[int, int]:
        # Metal's replaceRegion/getBytes require both to be 0 for a 1D
        # texture (there's no row/slice stride concept for it); bytes_per_image
        # is otherwise only meaningful for a 3D texture.
        if self.dims == 1:
            return 0, 0
        bytes_per_row   = self.width * self.channels * self.dtype.itemsize
        bytes_per_image = self.height * bytes_per_row if self.dims == 3 else 0
        return bytes_per_row, bytes_per_image

    def _check_data_shape(self, data: np.ndarray) -> None:
        if data.shape != self.shape:
            raise ValueError(f"Data shape {data.shape} does not match texture shape {self.shape}")

    def _check_same_device(self, other: "Buffer | Texture", kind: str) -> None:
        # Metal forbids referencing resources from different MTLDevices in
        # one command buffer -- same invariant Device._binary_op/_compare_op
        # enforce for Buffer-Buffer ops (device.py), extended here to the
        # new Texture<->Buffer/Texture<->Texture GPU-side paths. Without
        # this, a mismatched device crashes inside Metal's validation layer
        # instead of raising a catchable Python exception.
        if other._device is not self._device:
            raise ValueError(
                f"{kind} belongs to a different Device instance -- Metal does not "
                f"allow referencing resources from different MTLDevice objects in "
                f"the same command buffer"
            )

    def upload(self, data: np.ndarray) -> None:
        """Writes data into an internal Shared staging Buffer (sub-allocated
        from Device._staging_buffer()'s lazily-grown Heap -- see its
        docstring), then blits it into this texture via
        upload_from_buffer() -- instead of MTL::Texture::replaceRegion,
        which this project used before measuring that a Heap-staged blit
        beats it: a *fresh* standalone Buffer allocation pays a real
        first-write cost that scales with size (uncommitted physical
        pages faulting in), but this Device's staging Heap is committed
        once up front, so a fresh sub-allocation from it skips almost all
        of that cost -- measured ~1.9-3.4x faster than replaceRegion at
        1080p/4K (loses only at very small textures, ~480p and below,
        where replaceRegion's simplicity wins). Also means this now works
        on a private=True texture (replaceRegion can't touch Private
        storage at all; a blit doesn't care).

        The staging Buffer is only ever alive for the duration of one call
        (freed the instant this returns, since nothing else references it
        -- wait=True internally, so the blit has genuinely finished by
        then), so this Device's staging Heap only ever needs to hold one
        upload's worth of space, regardless of how many times you call
        this. For a hot loop, reusing your own Buffer via
        upload_from_buffer() directly is just as fast and skips the
        internal staging-buffer indirection; see its docstring, and
        Heap's class docstring, for when you'd want to manage this
        yourself instead (holding several independent results alive at
        once, which this method structurally can't help with)."""
        self._check_data_shape(data)
        arr = np.ascontiguousarray(data, dtype=self.dtype)
        buf = self._device._staging_buffer(arr.size, self.dtype)
        buf.contents[:] = arr.reshape(-1)
        self.upload_from_buffer(buf)

    def upload_from_buffer(self, buf: "Buffer", offset: int = 0, wait: bool = True) -> None:
        """Hardware-blit upload: copies buf's data into this texture via
        MTLBlitCommandEncoder (Device.blit_upload_texture() in the C++
        layer), instead of .upload()'s CPU-side replaceRegion copy. This
        texture keeps its normal (possibly tiled/swizzled) internal layout
        -- the blit engine retiles on the GPU side, concurrently with the
        CPU, rather than the CPU computing that conversion inline.

        buf must already hold this texture's data tightly packed (same
        convention .upload() expects from an ndarray: dtype matching this
        texture's per-channel dtype) starting at a byte offset into buf --
        write it there with a plain buf.contents[:] = ... first, an
        ordinary linear CPU memcpy. offset lets one larger buffer stage more
        than one texture's data (e.g. buf sized for two textures, the second
        uploaded via offset=first_texture_nbytes).

        For maximum throughput: reuse a single Buffer across many calls if
        you only need one result alive at a time (simplest, and just as
        fast as any alternative -- write new data into it, call this,
        repeat). If you need several independent Buffers alive
        concurrently, allocate them from a Device.heap() sized for your
        own concurrency instead of standalone Device.buffer()/.empty() --
        see Heap's class docstring for the measured win (~1.9-3.4x at
        1080p/4K) and why mtlpy doesn't do this for you automatically.

        A thin wrapper around Device.blit_upload_texture(buf, self, ...) --
        see that method if you'd rather call it directly on the Device."""
        self._device.blit_upload_texture(buf, self, offset=offset, wait=wait)

    def optimize_for_gpu_access(self, wait: bool = True) -> None:
        """Encodes MTLBlitCommandEncoder.optimizeContentsForGPUAccess --
        lets Metal repack this texture's contents into its preferred
        GPU-side layout after the fact. Only meaningful for a Shared-storage
        texture (self.is_private == False): a Private-storage texture
        already gets this automatically at creation per Apple's docs, so
        calling this on one is a redundant no-op. Contents must already be
        populated (upload()/upload_from_buffer()) before calling this."""
        self._device._dev.optimize_texture_for_gpu_access(self._tex, wait)

    def copy_to(self, dst: "Texture", wait: bool = True) -> None:
        """Hardware-blit texture-to-texture copy (MTLBlitCommandEncoder::
        copyFromTexture), the direct counterpart to upload_from_buffer()/
        to_buffer() for the Texture<->Texture case -- moving raw bytes on
        the GPU with no shader/format-conversion path involved, so unlike
        to_buffer() this works for any pixel format (Unorm included) and any
        combination of Shared/Private storage on either side. Useful for
        e.g. copying a Shared texture (populated via upload()) to a Private
        one (for Metal's more aggressive internal layout -- see
        Device.empty_texture()'s private= param) without a CPU round trip.

        dst must already exist with the same pixel_format and shape as
        self (create it with Device.empty_texture() first) -- this copies
        into an existing texture, it doesn't allocate one."""
        self._check_same_device(dst, "Destination texture")
        if dst.pixel_format != self.pixel_format:
            raise TypeError(
                f"Destination pixel_format {dst.pixel_format!r} doesn't match "
                f"source pixel_format {self.pixel_format!r}"
            )
        if dst.shape != self.shape:
            raise ValueError(
                f"Destination shape {dst.shape} doesn't match source shape {self.shape}"
            )
        self._device._dev.copy_texture(self._tex, dst._tex, wait)

    def download_to_buffer(self, buf: "Buffer", offset: int = 0, wait: bool = True) -> None:
        """Hardware-blit download: the read counterpart to
        upload_from_buffer() -- copies this texture's pixel data into buf
        via MTLBlitCommandEncoder, instead of .download()'s CPU-side
        getBytes() copy or to_buffer()'s compute-kernel readback. See
        Device.blit_download_texture()'s docstring for the
        concrete advantages over the latter (no compute dispatch, works
        even with readable=False, lands in a buffer you already own).

        buf must already have room for this texture's tightly packed bytes
        (dtype matching this texture's per-channel dtype) starting at a
        byte offset into buf -- same layout/offset contract as
        upload_from_buffer(), reversed. A thin wrapper around
        Device.blit_download_texture(self, buf, ...).

        For maximum throughput, same guidance as upload_from_buffer(): a
        single reused Buffer if you only need one result alive at a time,
        or a Device.heap() sized for your own concurrency if you need
        several independent results alive at once -- see Heap's class
        docstring for the measured win (~1.5-2.4x) and why this isn't
        automatic inside .download()."""
        self._device.blit_download_texture(self, buf, offset=offset, wait=wait)

    def download(self) -> np.ndarray:
        """Hardware blit into an internal Shared staging Buffer
        (sub-allocated from Device._staging_buffer()'s lazily-grown Heap --
        see its docstring), then a plain CPU copy out of that Buffer's
        .numpy() into an ordinary, independent array -- instead of
        MTL::Texture::getBytes(), which this project used before measuring
        that a Heap-staged blit (even with the extra copy-out) beats it:
        getBytes() is a real CPU-bound copy that scales with image size,
        while a blit's cost is mostly fixed per-call overhead that barely
        grows with size -- measured ~1.2-1.5x faster at every size tested
        (net of the copy-out; the copy costs real time, see this method's
        history for numbers without it, but doesn't erase the win). Also
        works regardless of this texture's storage mode, including
        private=True -- getBytes() can't touch Private storage at all.

        The copy-out (not a zero-copy view, unlike Buffer.numpy() and
        unlike a naive "just return the staging Buffer's .numpy()") is
        deliberate: it's what lets this Device's staging Heap stay small
        and fixed regardless of how many .download() results you hold
        onto simultaneously, since each call's staging Buffer is provably
        unreferenced (and so immediately freed, and its heap space
        reclaimed) the instant this returns -- a *reused* result-holding
        Buffer would instead need capacity for however many results are
        alive at once, which only the caller can size correctly (see
        Heap's class docstring for that case, via download_to_buffer())."""
        n = utils.shape_size(self.shape)
        buf = self._device._staging_buffer(n, self.dtype)
        self.download_to_buffer(buf)
        return buf.numpy().reshape(self.shape).copy()

    def numpy(self) -> np.ndarray:
        """Alias for .download() -- see the class docstring for why this
        (unlike Buffer.numpy()) is always a real copy, not a view."""
        return self.download()

    def to_buffer(self) -> "Buffer":
        """GPU-side readback into a tightly packed Buffer (see
        Device.buffer_from_texture()), instead of the CPU-side getBytes()
        copy .download()/.numpy() use -- the result's .contents/.numpy()
        are genuinely zero-copy, same as any other Buffer. Requires
        self.readable (raises otherwise -- a texture created with
        readable=False lacks the MTLTextureUsageShaderRead the copy kernel
        needs)."""
        return self._device.buffer_from_texture(self)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        # Unlike Buffer, .numpy() always makes a real copy (see class
        # docstring) -- there's no way to satisfy a no-copy request at all.
        if copy is False:
            raise ValueError(
                "Texture data isn't directly addressable from the CPU (see "
                "the class docstring), so copy=False can never be satisfied"
            )
        arr = self.numpy()
        return arr if dtype is None or np.dtype(dtype) == arr.dtype else arr.astype(dtype)

    def __repr__(self) -> str:
        private = ", private" if self.is_private else ""
        return (f"Texture(shape={self.shape}, pixel_format={self.pixel_format!r}, "
                f"dtype={self.dtype}{private})")


class Sampler:
    """A Metal sampler state for texture sampling kernels (access::sample
    in MSL) -- see Device.sampler() and shader.texture_type()."""

    def __init__(self, _sampler, linear: bool, repeat: bool, device):
        self._sampler = _sampler   # _mtlpy.Sampler
        self.linear   = linear
        self.repeat   = repeat
        self._device  = device

    def __repr__(self) -> str:
        return f"Sampler(linear={self.linear}, repeat={self.repeat})"
