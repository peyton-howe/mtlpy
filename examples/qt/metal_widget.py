"""A PySide6 QWidget that displays mtlpy Textures via mtlpy.Surface.

Example code, not part of the mtlpy package -- copy it into your project
(or import it from here, as the other Qt examples do) and adapt as needed.
It's the Qt-specific glue around Surface: a native child NSView, no Qt
painting, and keeping the layer in sync on resize / screen change.

    from PySide6.QtWidgets import QApplication
    from metal_widget import MetalWidget

    app = QApplication([])
    view = MetalWidget(device)
    view.resize(800, 600)
    view.show()
    ...
    view.present(tex)        # e.g. from a QTimer callback

Requires PySide6 (pip install PySide6).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QWidget

from mtlpy import Frame

if TYPE_CHECKING:
    from mtlpy import Device, Surface, Texture

# Events after which the layer's frame/backing scale may be stale.
_LAYOUT_EVENTS = {QEvent.Type.Show, QEvent.Type.ScreenChangeInternal}
if hasattr(QEvent.Type, "DevicePixelRatioChange"):  # Qt >= 6.6
    _LAYOUT_EVENTS.add(QEvent.Type.DevicePixelRatioChange)


class MetalWidget(QWidget):
    """A native QWidget backed by an mtlpy Surface (CAMetalLayer).

    Zero-copy rendering: kernels write straight into the window's drawable.

        frame = view.next_frame()       # None while hidden: skip the frame
        if frame is not None:
            with frame:                 # presents at the end of the block
                kernel.run([], grid=(frame.width, frame.height, 1),
                           textures=[frame.texture], wait=False)

    present(texture) also works for an existing Texture (one GPU copy pass).

    Qt does no painting of its own here -- the widget's content is whatever
    was last presented, scaled to the widget per the Surface's
    scaling option. The last frame stays on screen until the next present,
    including across expose/resize, so there's no need to re-present from
    paintEvent.

    surface_options are forwarded to Surface (pixel_format, colorspace, edr,
    scaling, filter, display_sync, max_drawables).

    Must be created on the GUI (main) thread. present() may be called from
    any thread, but skips the frame while the widget is hidden to avoid
    CAMetalLayer's 1s nextDrawable stall."""

    def __init__(self, device: Device, parent: QWidget | None = None, **surface_options):
        super().__init__(parent)
        # Own NSView (winId() of a non-native child would be the window's),
        # and no Qt raster painting/background fill underneath the layer.
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._surface = device.surface(int(self.winId()), **surface_options)

    @property
    def surface(self) -> Surface:
        return self._surface

    def next_frame(self, size: tuple[int, int] | None = None) -> Frame | None:
        """Surface.next_frame(size) -- zero-copy: render into the returned
        frame.texture, then present(frame). None (skip the frame) while the
        widget isn't visible."""
        if not self.isVisible():
            return None
        return self._surface.next_frame(size)

    def present(self, source: Texture | Frame, **kwargs) -> bool:
        """Surface.present(source, **kwargs). A Texture is skipped (returns
        False) while the widget isn't visible; a Frame is always presented."""
        if not isinstance(source, Frame) and not self.isVisible():
            return False
        return self._surface.present(source, **kwargs)

    # -- Qt overrides ---------------------------------------------------

    def paintEngine(self):
        return None  # with WA_PaintOnScreen: Qt never paints this widget

    def paintEvent(self, event) -> None:
        pass  # the CAMetalLayer keeps showing the last presented frame

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._surface.update_layout()

    def event(self, event) -> bool:
        if event.type() in _LAYOUT_EVENTS:
            self._surface.update_layout()
        return super().event(event)
