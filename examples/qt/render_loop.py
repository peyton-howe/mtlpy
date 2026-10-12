"""Animate a compute kernel in a PySide6 window -- zero-copy.

Each frame the kernel writes straight into the window's CAMetalLayer
drawable (frame.texture), which is then presented as-is: no intermediate
texture, no copy pass, no CPU round-trip. The frame is sized to the widget
in pixels, so the kernel renders at native (Retina) resolution.

Nothing waits on the GPU: the kernel and the present both go to the
Device's default queue (so the present is ordered after the kernel), and
with vsync on (the default) next_frame() paces the loop to the display.

Requires PySide6 (pip install PySide6). Uses metal_widget.py from this
directory -- run from anywhere: python examples/qt/render_loop.py
"""
import time

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import mtlpy
from metal_widget import MetalWidget  # examples/qt/metal_widget.py

source = """
#include <metal_stdlib>
using namespace metal;
kernel void plasma(
    device const float* t [[buffer(0)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Normalized by height so the pattern keeps its aspect on resize.
    float2 uv = float2(gid) / float(out.get_height());
    float v = sin(uv.x * 10.0 + t[0])
            + sin((uv.y * 10.0 + t[0]) * 0.5)
            + sin(length(uv - 0.5) * 20.0 - t[0] * 2.0);
    float3 c = 0.5 + 0.5 * cos(v + float3(0.0, 2.0, 4.0));
    out.write(float4(c, 1.0), gid);   // RGBA; Metal stores it as BGRA
}
"""

app = QApplication([])
device = mtlpy.Device()
plasma = device.compile(source, "plasma")

# One time-uniform buffer per in-flight frame (max_drawables=3), so the CPU
# never overwrites a value a still-running dispatch is reading.
times = [device.buffer(np.zeros(1, np.float32)) for _ in range(3)]

view = MetalWidget(device)
view.setWindowTitle("mtlpy + PySide6 (zero-copy)")
view.resize(960, 540)
view.show()

start = time.perf_counter()
frame_index = 0

def tick():
    global frame_index
    frame = view.next_frame()
    if frame is None:   # hidden / minimized / occluded: skip
        return
    with frame:         # presents at the end of the block
        t = times[frame_index % len(times)]
        t.contents[0] = time.perf_counter() - start
        plasma.run([t], grid=(frame.width, frame.height, 1),
                   textures=[frame.texture], wait=False)
    frame_index += 1

timer = QTimer()
timer.timeout.connect(tick)
timer.start(0)  # as fast as next_frame() (vsync) allows
app.exec()
