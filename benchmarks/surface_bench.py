"""Zero-copy (next_frame -> kernel writes drawable -> present(frame)) vs.
copy (kernel writes a Texture -> present(texture), one extra GPU pass).

Wall-clock frame time isn't a useful comparison here: presents are paced by
the compositor at the display's refresh rate even with vsync off, so both
modes measure ~1/refresh. Instead this compares GPU execution time from
Metal's command buffer timestamps: the fill kernel itself, and the GPU work
each present adds on top (Surface.last_present_gpu_time).

    python benchmarks/surface_bench.py
"""
import statistics
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

import mtlpy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "qt"))
from metal_widget import MetalWidget  # noqa: E402

FRAMES = 200
SIZES = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]

SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void fill(texture2d<float, access::write> t [[texture(0)]],
                 uint2 gid [[thread_position_in_grid]]) {
    float2 uv = float2(gid) / float2(t.get_width(), t.get_height());
    t.write(float4(uv, 0.5 + 0.5 * sin(uv.x * 40.0), 1.0), gid);
}
"""


def measure(view, fill, w, h, zero_copy, tex=None):
    kernel_ms, present_ms = [], []
    for i in range(FRAMES + 10):
        if zero_copy:
            frame = view.next_frame((w, h))
            if frame is None:
                raise RuntimeError(f"Surface benchmark: no drawable available for {w}x{h}")
            target = frame.texture
        else:
            target = tex
        start, end = fill.run([], grid=(w, h, 1), textures=[target], wait=True)
        view.present(frame if zero_copy else tex, wait=True)
        if i >= 10:  # warm-up
            kernel_ms.append((end - start) * 1e3)
            present_ms.append(view.surface.last_present_gpu_time * 1e3)
    return statistics.median(kernel_ms), statistics.median(present_ms)


def main():
    app = QApplication([])
    device = mtlpy.Device()
    fill = device.compile(SOURCE, "fill")
    view = MetalWidget(device, display_sync=False)
    view.resize(960, 540)
    view.show()
    app.processEvents()

    print(f"GPU ms per frame, median of {FRAMES}\n")
    print(f"{'size':>11}  {'kernel':>7}  {'copy: present':>14}  {'zero-copy: present':>19}  "
          f"{'copy overhead':>13}")
    for w, h in SIZES:
        tex = device.empty_texture((h, w), "bgra8Unorm")
        k_copy, p_copy = measure(view, fill, w, h, zero_copy=False, tex=tex)
        k_zc, p_zc = measure(view, fill, w, h, zero_copy=True)
        k = (k_copy + k_zc) / 2
        print(f"{w:>5}x{h:<5}  {k:>7.3f}  {p_copy:>14.3f}  {p_zc:>19.3f}  "
              f"{(p_copy - p_zc) / k * 100:>12.0f}%")
        app.processEvents()
    print("\ncopy overhead = extra GPU time present(texture) adds, relative to the kernel")


if __name__ == "__main__":
    main()
