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

    def download(self) -> np.ndarray:
        bytes_per_row, bytes_per_image = self._bytes_per_row_and_image()
        nbytes = utils.shape_size(self.shape) * self.dtype.itemsize
        raw = self._tex.download(nbytes, bytes_per_row, bytes_per_image)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape)

    def numpy(self) -> np.ndarray:
        """Alias for .download() -- see the class docstring for why this
        (unlike Buffer.numpy()) is always a real copy, not a view."""
        return self.download()

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
        return (f"Texture(shape={self.shape}, pixel_format={self.pixel_format!r}, "
                f"dtype={self.dtype})")


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
