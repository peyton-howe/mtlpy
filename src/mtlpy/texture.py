from __future__ import annotations
import numpy as np
from . import utils


class Texture:
    """A Metal texture (1D/2D/3D). Unlike Buffer, a Texture's CPU-visible
    memory layout isn't guaranteed to be a tightly packed array (Metal may
    pad/tile rows internally), so there's no Buffer.contents equivalent --
    .upload()/.download() are genuine copies via MTL::Texture's
    replaceRegion/getBytes, not a live view over GPU memory."""

    def __init__(self, _tex, dims: int, pixel_format: str,
                 width: int, height: int, depth: int, device):
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
        self._device          = device

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

    def upload(self, data: np.ndarray) -> None:
        if data.shape != self.shape:
            raise ValueError(f"Data shape {data.shape} does not match texture shape {self.shape}")
        arr = np.ascontiguousarray(data, dtype=self.dtype)
        bytes_per_row, bytes_per_image = self._bytes_per_row_and_image()
        self._tex.upload(arr, bytes_per_row, bytes_per_image)

    def upload_fast(self, data: np.ndarray, wait: bool = True) -> None:
        """Convenience wrapper around upload_from_buffer(): stages data into
        a fresh Buffer (a plain CPU memcpy into Buffer.contents, always
        fast/zero-copy) and blit-uploads that into this texture, instead of
        .upload()'s CPU-side replaceRegion copy -- measured up to ~9x faster
        at 4K, and unlike .upload(), works on a private=True texture too.
        Allocates a new staging Buffer on every call; for a hot loop
        re-uploading to the same texture repeatedly, allocate your own
        Buffer once and call upload_from_buffer() directly instead (same
        allocate-once tradeoff as Buffer's own out-of-place convenience ops
        -- see the README's "Reusing buffers in a hot loop")."""
        if data.shape != self.shape:
            raise ValueError(f"Data shape {data.shape} does not match texture shape {self.shape}")
        arr = np.ascontiguousarray(data, dtype=self.dtype)
        buf = self._device.buffer(arr)
        self.upload_from_buffer(buf, wait=wait)

    def upload_from_buffer(self, buf: "Buffer", offset: int = 0, wait: bool = True) -> None:
        """Hardware-blit upload: copies buf's data into this texture via
        MTLBlitCommandEncoder (Device.blit_upload_texture() in the C++
        layer), instead of .upload()'s CPU-side replaceRegion copy. This
        texture keeps its normal (possibly tiled/swizzled) internal layout
        -- the blit engine retiles on the GPU side, concurrently with the
        CPU, rather than the CPU computing that conversion inline.

        buf must already hold this texture's data tightly packed (same
        convention .upload() expects from an ndarray: dtype matching this
        texture's per-channel dtype, element count matching
        shape_size(self.shape)) -- write it there with a plain
        buf.contents[:] = ... first, an ordinary linear CPU memcpy. offset
        is a byte offset into buf, for reusing one buffer to stage more
        than one texture's data."""
        expected_elements = utils.shape_size(self.shape)
        if buf.dtype != self.dtype:
            raise TypeError(
                f"Buffer dtype {buf.dtype} doesn't match texture pixel_format "
                f"{self.pixel_format!r}'s per-channel dtype {self.dtype}"
            )
        if buf.size != expected_elements:
            raise ValueError(
                f"Buffer has {buf.size} elements, but this {self.shape} texture "
                f"needs {expected_elements}"
            )
        bytes_per_row, bytes_per_image = self._bytes_per_row_and_image()
        self._device._dev.blit_upload_texture(
            buf._buf, offset, self._tex, bytes_per_row, bytes_per_image, wait)

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

    def download(self) -> np.ndarray:
        bytes_per_row, bytes_per_image = self._bytes_per_row_and_image()
        nbytes = utils.shape_size(self.shape) * self.dtype.itemsize
        raw = self._tex.download(nbytes, bytes_per_row, bytes_per_image)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape)

    def numpy(self) -> np.ndarray:
        """Alias for .download() -- see the class docstring for why this
        (unlike Buffer.numpy()) is always a real copy, not a view."""
        return self.download()

    def download_fast(self) -> np.ndarray:
        """Convenience wrapper around to_buffer(): reads this texture back
        via a GPU-side compute-kernel copy into a Buffer, then that Buffer's
        already-zero-copy .numpy(), instead of .download()'s CPU-side
        getBytes() copy -- measured ~1.5-1.6x faster at 1080p/4K, and unlike
        .download(), works on a private=True texture too. Inherits
        to_buffer()'s one restriction: raises NotImplementedError for a
        normalized (Unorm) pixel_format -- use .download() for those."""
        return self.to_buffer().numpy()

    def to_buffer(self) -> "Buffer":
        """GPU-side readback into a tightly packed Buffer (see
        Device.buffer_from_texture()), instead of the CPU-side getBytes()
        copy .download()/.numpy() use -- the result's .contents/.numpy()
        are genuinely zero-copy, same as any other Buffer."""
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
