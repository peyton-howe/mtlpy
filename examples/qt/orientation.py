"""Visual check that both display paths are upright, unmirrored, and have
the right channel order.

Draws a test card with QPainter (readable text, an arrow, and one solid color
per corner), uploads it as an rgba8Unorm Texture, and alternates every 2s
between the two ways of getting it on screen:

- copy:      surface.present(texture)
- zero-copy: a kernel samples the texture straight into frame.texture

The window title says which path is showing. Both should look identical and
match the card exactly:

    RED  top-left        GREEN top-right
    BLUE bottom-left     WHITE bottom-right
    text reads left-to-right, arrow points up

A vertical flip swaps the top and bottom corners. A mirror reverses the
text. An R/B swap (BGRA confusion) turns red into blue.

Requires PySide6 (pip install PySide6). Run: python examples/qt/orientation.py
"""
import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

import mtlpy
from metal_widget import MetalWidget  # examples/qt/metal_widget.py

W, H = 800, 500
CORNER = 120

blit_source = """
#include <metal_stdlib>
using namespace metal;
kernel void blit(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    sampler                          smp [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    dst.write(src.sample(smp, uv), gid);   // RGBA in; Metal stores BGRA
}
"""


def test_card() -> np.ndarray:
    img = QImage(W, H, QImage.Format.Format_RGBA8888)
    img.fill(QColor(40, 40, 40))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.fillRect(0, 0, CORNER, CORNER, QColor(255, 0, 0))                    # top-left
    p.fillRect(W - CORNER, 0, CORNER, CORNER, QColor(0, 255, 0))           # top-right
    p.fillRect(0, H - CORNER, CORNER, CORNER, QColor(0, 0, 255))           # bottom-left
    p.fillRect(W - CORNER, H - CORNER, CORNER, CORNER, QColor(255, 255, 255))  # bottom-right

    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont("Helvetica", 22, QFont.Weight.Bold))
    p.drawText(QRect(CORNER + 10, 10, W - 2 * CORNER - 20, 40),
               Qt.AlignmentFlag.AlignLeft, "TOP-LEFT: red")
    p.drawText(QRect(CORNER + 10, H - 50, W - 2 * CORNER - 20, 40),
               Qt.AlignmentFlag.AlignRight, "bottom-right: white")
    p.setFont(QFont("Helvetica", 64, QFont.Weight.Bold))
    p.drawText(QRect(0, H // 2 - 10, W, 90), Qt.AlignmentFlag.AlignCenter, "mtlpy  ABC 123")

    # Arrow pointing up.
    pen = QPen(QColor(255, 200, 0), 12)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    cx, top, bottom = W // 2, 70, H // 2 - 30
    p.drawLine(cx, bottom, cx, top)
    p.drawLine(cx, top, cx - 40, top + 40)
    p.drawLine(cx, top, cx + 40, top + 40)
    p.end()

    rows = np.frombuffer(img.constBits(), np.uint8).reshape(H, img.bytesPerLine())
    return rows[:, : W * 4].reshape(H, W, 4).copy()   # row 0 = top of the card


app = QApplication([])
device = mtlpy.Device()
card = device.texture(test_card(), "rgba8Unorm")
blit = device.compile(blit_source, "blit")
linear = device.sampler(linear=True)

view = MetalWidget(device)
view.resize(W, H)
view.show()

mode = {"zero_copy": False}


def draw():
    if mode["zero_copy"]:
        frame = view.next_frame()
        if frame is None:
            return
        with frame:
            blit.run([], grid=(frame.width, frame.height, 1),
                     textures=[card, frame.texture], samplers=[linear], wait=False)
    else:
        view.present(card)


def toggle():
    mode["zero_copy"] = not mode["zero_copy"]
    name = "zero-copy: next_frame()" if mode["zero_copy"] else "copy: present(texture)"
    view.setWindowTitle(f"mtlpy orientation check -- {name}")
    draw()


toggle()
redraw = QTimer()
redraw.timeout.connect(draw)       # keeps zero-copy frames sized to the window
redraw.start(16)
flip = QTimer()
flip.timeout.connect(toggle)
flip.start(2000)
app.exec()
