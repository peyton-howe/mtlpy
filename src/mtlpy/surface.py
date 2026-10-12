from __future__ import annotations
from typing import TYPE_CHECKING

from .texture import Texture

if TYPE_CHECKING:
    from .device import Device
    from .sync import Event


# Pixel formats CAMetalLayer accepts for its drawables. Deliberately separate
# from utils' texture table: the 10-bit formats are packed (no per-channel
# numpy dtype), and the drawable is never uploaded/downloaded anyway.
_DRAWABLE_FORMATS = {
    "bgra8Unorm":      80,
    "bgra8Unorm_srgb": 81,
    "rgb10a2Unorm":    90,
    "bgr10a2Unorm":    94,
    "rgba16Float":     115,
}

# Drawable formats a Frame's texture can be wrapped as an mtlpy Texture in
# (the 10-bit ones are packed -- no per-channel dtype in utils' table).
_ZERO_COPY_FORMATS = {"bgra8Unorm", "bgra8Unorm_srgb", "rgba16Float"}

# Must match Surface::Gravity / Surface::Colorspace in csrc/surface.h.
_SCALING = {"aspect": 0, "fill": 1, "stretch": 2, "center": 3}
_COLORSPACES = {
    None: 0,
    "srgb": 1,
    "display_p3": 2,
    "linear_srgb": 3,
    "extended_linear_srgb": 4,
}


class Frame:
    """One drawable of a Surface, for zero-copy display: write into
    frame.texture with your kernels, then surface.present(frame). The texture
    *is* the image that gets composited -- nothing is copied.

        frame = surface.next_frame()
        if frame is not None:
            kernel.run([], grid=(frame.width, frame.height, 1),
                       textures=[frame.texture], wait=False)
            surface.present(frame)

    or, presenting automatically at the end of the block:

        if frame is not None:
            with frame:
                kernel.run(..., textures=[frame.texture], wait=False)

    frame.texture is in the Surface's pixel_format (bgra8Unorm by default --
    kernels still write float4 RGBA; Metal does the BGRA swizzle on store),
    GPU-private, and only valid until presented: afterwards Core Animation
    owns it and recycles it for a later frame. Its contents on acquisition
    are undefined (often a stale earlier frame), so write every pixel.

    A Frame holds one of the layer's max_drawables drawables until it's
    presented or garbage collected -- don't keep them around."""

    def __init__(self, _frame, surface: Surface):
        self._frame = _frame  # _mtlpy.Frame
        self._surface = surface
        self.width = _frame.width
        self.height = _frame.height
        self.pixel_format = surface.pixel_format
        self._texture: Texture | None = None

    @property
    def texture(self) -> Texture:
        if self._frame.presented:
            raise RuntimeError("Frame was already presented")
        if self._texture is None:
            self._texture = Texture(
                self._frame.texture(), 2, self.pixel_format, self.width,
                self.height, 1, self._surface._device,
                readable=True, writable=True)
        return self._texture

    @property
    def presented(self) -> bool:
        return self._frame.presented

    def __enter__(self) -> Frame:
        return self

    def __exit__(self, exc_type, *exc_info) -> None:
        if exc_type is None and not self.presented:
            self._surface.present(self)


class Surface:
    """Displays Textures in a native macOS view through a CAMetalLayer.

    Created with Device.surface(view_ptr) from any NSView* -- e.g.
    int(widget.winId()) of a native PySide6 QWidget; examples/qt/metal_widget.py
    is a ready-made QWidget around it.

        surface = device.surface(int(widget.winId()))
        pipeline.run([], grid=(w, h, 1), textures=[tex])   # no wait needed
        surface.present(tex)

    Zero-copy: next_frame() returns the layer's next drawable as a Frame
    whose .texture your kernels write directly; present(frame) then presents
    it with no copy at all. See Frame. Its texture is sized to the view's
    pixels by default and is in the drawable pixel_format.

    present(texture) instead draws an existing Texture into the next drawable
    and presents it -- one GPU pass, for textures that must persist across
    frames, have their own size, or are in a format a drawable can't be. The drawable is sized to the texture and Core Animation scales it to
    the view (see scaling), so the texture doesn't need to follow window
    resizes or Retina scale. Any float-readable 2D format works (rgba8Unorm,
    bgra8Unorm, r32Float, rgba16Float, ...): 1-channel textures display as
    grayscale, 2-channel as (r, g, 0). Integer formats must be converted
    first. Alpha is ignored (the layer is opaque).

    Ordering: present() encodes on the Device's default queue -- the same one
    Pipeline.run() uses -- so it's ordered after any dispatch already
    committed there, wait=False or not. For work produced on a secondary
    Queue, pass wait_for=(event, value) with an Event that work signals.

    Threading: construction, update_layout() and close() must run on the
    main thread (AppKit). present() can run from any thread and releases
    the GIL.

    Args:
        pixel_format: drawable format -- one of bgra8Unorm (default),
            bgra8Unorm_srgb, rgb10a2Unorm, bgr10a2Unorm, rgba16Float.
        colorspace: how the compositor interprets drawable values -- "srgb"
            (default), "display_p3", "linear_srgb", "extended_linear_srgb",
            or None for untagged.
        edr: request extended dynamic range output (values > 1.0); pair with
            pixel_format="rgba16Float", colorspace="extended_linear_srgb".
        scaling: "aspect" (fit, letterboxed; default), "fill" (cover,
            cropped), "stretch", or "center" (1:1 texels, no scaling).
        filter: "linear" (default) or "nearest" (crisp pixels when zoomed).
        display_sync: vsync. False presents immediately (may tear).
        max_drawables: 2 or 3 (default) -- lower latency vs. throughput.
    """

    def __init__(self, device: Device, view_ptr: int, *,
                 pixel_format: str = "bgra8Unorm",
                 colorspace: str | None = "srgb",
                 edr: bool = False,
                 scaling: str = "aspect",
                 filter: str = "linear",
                 display_sync: bool = True,
                 max_drawables: int = 3):
        if pixel_format not in _DRAWABLE_FORMATS:
            raise ValueError(
                f"Unsupported surface pixel_format '{pixel_format}'. "
                f"Supported: {sorted(_DRAWABLE_FORMATS)}")
        if scaling not in _SCALING:
            raise ValueError(f"scaling must be one of {sorted(_SCALING)}, got '{scaling}'")
        if colorspace not in _COLORSPACES:
            raise ValueError(
                f"colorspace must be one of {sorted(k for k in _COLORSPACES if k)} "
                f"or None, got '{colorspace}'")
        if filter not in ("linear", "nearest"):
            raise ValueError(f"filter must be 'linear' or 'nearest', got '{filter}'")
        if not view_ptr:
            raise ValueError("view_ptr must be a non-null NSView*")

        self._device = device
        self.pixel_format = pixel_format
        self._surf = device._dev.create_surface(
            int(view_ptr), _DRAWABLE_FORMATS[pixel_format], _SCALING[scaling],
            filter == "linear", _COLORSPACES[colorspace], edr, display_sync,
            max_drawables)

    def next_frame(self, size: tuple[int, int] | None = None) -> Frame | None:
        """Acquire the next drawable for zero-copy rendering (see Frame).

        size is (width, height) in pixels; default is the view's current
        size in pixels, i.e. one texel per screen pixel. Other sizes are
        scaled to the view per the scaling option.

        Returns None (skip this frame) if the view has no area, or if no
        drawable became available within CAMetalLayer's 1s timeout (e.g. the
        window is occluded). May block until a drawable is free -- with
        display_sync that paces the caller to the display. Releases the GIL
        while waiting."""
        if self._surf is None:
            raise RuntimeError("Surface is closed")
        if self.pixel_format not in _ZERO_COPY_FORMATS:
            raise ValueError(
                f"next_frame() needs pixel_format in {sorted(_ZERO_COPY_FORMATS)}, "
                f"this Surface uses '{self.pixel_format}' -- use present(texture)")
        w, h = (0, 0) if size is None else (int(size[0]), int(size[1]))
        if size is not None and (w <= 0 or h <= 0):
            raise ValueError(f"size must be positive, got {size}")
        raw = self._surf.next_frame(w, h)
        return None if raw is None else Frame(raw, self)

    def present(self, source: Texture | Frame, *, wait: bool = False,
                wait_for: tuple[Event, int] | None = None) -> bool:
        """Present a Frame from next_frame() (zero-copy), or draw a Texture
        into the next drawable and present that.

        Presenting is ordered after GPU work already committed to the
        Device's default queue, so kernels that wrote the source need no
        wait first.

        Returns False if a Texture's frame was skipped because no drawable
        became available within CAMetalLayer's 1s timeout (e.g. the window
        is hidden or occluded), True otherwise. (A Frame already holds its
        drawable, so presenting one always returns True.)

        wait=True blocks until the GPU has finished drawing (mostly useful
        for tests). wait_for=(event, value) makes the GPU wait for event to
        reach value before reading texture -- needed only when texture was
        written on a secondary Queue."""
        if self._surf is None:
            raise RuntimeError("Surface is closed")
        event, value = (wait_for[0]._event, int(wait_for[1])) if wait_for else (None, 0)
        if isinstance(source, Frame):
            if source._surface is not self:
                raise ValueError("Frame belongs to a different Surface")
            if source.presented:
                raise RuntimeError("Frame was already presented")
            self._surf.present_frame(source._frame, wait, event, value)
            return True
        texture = source
        if texture._device is not self._device:
            raise ValueError("Texture belongs to a different Device")
        if texture.dims != 2:
            raise ValueError(f"Surface.present needs a 2D texture, got {texture.dims}D")
        if not texture.readable:
            raise ValueError("Surface.present needs a texture created with readable=True")
        if texture.msl_scalar_type == "uint":
            raise ValueError(
                f"Can't display integer texture format '{texture.pixel_format}' "
                f"directly -- convert to a normalized/float format first")
        return self._surf.present(texture._tex, wait, event, value)

    def update_layout(self) -> None:
        """Re-sync the layer to the view's current bounds and backing scale.
        Call from the view's resize / screen-change handler (main thread).
        MetalWidget does this for you."""
        if self._surf is not None:
            self._surf.update_layout()

    @property
    def last_present_gpu_time(self) -> float:
        """GPU execution time in seconds of the most recently completed
        present -- the copy pass for present(texture), ~0 for a zero-copy
        present(frame). Updated asynchronously when each present's command
        buffer completes (exact after present(..., wait=True))."""
        return 0.0 if self._surf is None else self._surf.last_present_gpu_time

    @property
    def view_size(self) -> tuple[int, int]:
        """(width, height) of the view in pixels as of the last
        update_layout() -- next_frame()'s default size."""
        if self._surf is None:
            return (0, 0)
        return (self._surf.view_width, self._surf.view_height)

    @property
    def drawable_size(self) -> tuple[int, int]:
        """(width, height) of the drawable in pixels -- the size of the last
        frame acquired or texture presented, or (0, 0) before the first one /
        after close."""
        if self._surf is None:
            return (0, 0)
        return (self._surf.drawable_width, self._surf.drawable_height)

    @property
    def closed(self) -> bool:
        return self._surf is None

    def close(self) -> None:
        """Detach the layer from the view and release its GPU state.
        Idempotent. Also happens automatically when the Surface is garbage
        collected, but call it (or use `with`) before the view goes away to
        be deterministic."""
        if self._surf is not None:
            self._surf.close()
            self._surf = None

    def __enter__(self) -> Surface:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
