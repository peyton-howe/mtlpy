"""Surface / CAMetalLayer presentation, via examples/qt/metal_widget.py's
PySide6 MetalWidget (so the example is exercised too).

Skipped when PySide6 isn't installed. Needs a GUI session (a window server),
which local macOS runs and GitHub's macOS runners both have.
"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import numpy as np
import pytest

import mtlpy

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "qt"))
from metal_widget import MetalWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def device():
    return mtlpy.Device()


def _destroy(app, w):
    # deleteLater() alone never runs outside a real event loop --
    # processEvents() doesn't deliver DeferredDelete -- which would keep the
    # widget (and its Surface/Device) alive until interpreter shutdown.
    w.surface.close()
    w.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def widget(app, device):
    w = MetalWidget(device)
    w.resize(320, 240)
    w.show()
    app.processEvents()
    yield w
    _destroy(app, w)


def _rgba(h=120, w=160):
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = 255
    img[..., 3] = 255
    return img


@pytest.mark.parametrize("fmt,data", [
    ("rgba8Unorm",  _rgba()),
    ("bgra8Unorm",  _rgba()),
    ("rgba16Float", _rgba().astype(np.float16) / 255),
    ("r32Float",    np.random.rand(120, 160).astype(np.float32)),
    ("r8Unorm",     np.zeros((120, 160), np.uint8)),
])
def test_present_formats(widget, device, fmt, data):
    tex = device.texture(data, fmt)
    assert widget.present(tex, wait=True)
    assert widget.surface.drawable_size == (160, 120)


def test_drawable_follows_texture_size(widget, device):
    for h, w in [(64, 64), (300, 500), (17, 33)]:
        assert widget.present(device.texture(_rgba(h, w), "rgba8Unorm"), wait=True)
        assert widget.surface.drawable_size == (w, h)


def test_present_after_kernel_no_wait(widget, device):
    # Same-queue ordering: no wait between the dispatch and the present.
    k = device.compile("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void fill(texture2d<float, access::write> t [[texture(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
            t.write(float4(0, 1, 0, 1), gid);
        }""", "fill")
    tex = device.empty_texture((90, 120), "rgba8Unorm")
    for _ in range(10):
        k.run([], grid=(120, 90, 1), textures=[tex], wait=False)
        assert widget.present(tex)
    assert widget.present(tex, wait=True)


def _pending_producer(device):
    ev = device.shared_event()
    result = device.buffer(np.zeros(1, dtype=np.uint32))
    marker = device.compile("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void mark(device uint* result [[buffer(0)]]) {
            result[0] = 42;
        }""", "mark")
    producer = device.command_buffer(queue=device.queue())
    marker.run([result], 1, cb=producer)
    return producer, ev, result


def _present_waiting_for_producer(widget, source, producer, ev):
    started = threading.Event()

    def present():
        assert ev.signaled_value == 0
        started.set()
        return widget.present(source, wait=True, wait_for=(ev, 1))

    # Submit the producer only after presentation has started with a pending
    # event. A missing GPU wait must not pass just because the producer won
    # a scheduling race or a readback later synchronized the default queue.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(present)
        try:
            assert started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.1)
        finally:
            producer.commit(wait=False)
        assert pending.result(timeout=5)
        assert ev.signaled_value == 1


def test_present_waits_for_event(widget, device, fill):
    tex = device.texture(_rgba(), "rgba8Unorm")
    producer, ev, result = _pending_producer(device)
    fill.run([], grid=(tex.width, tex.height, 1), textures=[tex], cb=producer)
    producer.signal_event(ev, 1)
    _present_waiting_for_producer(widget, tex, producer, ev)
    assert result.contents[0] == 42


def test_private_texture(widget, device):
    tex = device.empty_texture((64, 64), "rgba8Unorm", private=True)
    assert widget.present(tex, wait=True)


def test_present_while_hidden_skips(widget, device):
    widget.hide()
    assert widget.present(device.texture(_rgba(), "rgba8Unorm")) is False


def test_render_thread_visibility(app, widget, device, monkeypatch):
    tex = device.texture(_rgba(), "rgba8Unorm")

    def reject_visibility_query():
        pytest.fail("render methods must not query QWidget visibility")

    monkeypatch.setattr(widget, "isVisible", reject_visibility_query)

    def render(visible):
        assert widget.present(tex, wait=True) is visible
        frame = widget.next_frame()
        assert (frame is not None) is visible
        if frame is not None:
            assert widget.present(frame, wait=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        for visible in (True, False, True):
            widget.setVisible(visible)
            app.processEvents()
            executor.submit(render, visible).result(timeout=5)


def test_resize_keeps_presenting(app, widget, device):
    tex = device.texture(_rgba(), "rgba8Unorm")
    for size in [(640, 480), (100, 50), (320, 240)]:
        widget.resize(*size)
        app.processEvents()
        assert widget.present(tex, wait=True)


def test_rejects_integer_texture(widget, device):
    with pytest.raises(ValueError, match="integer"):
        widget.present(device.texture(_rgba(), "rgba8Uint"))


def test_rejects_unreadable_texture(widget, device):
    tex = device.empty_texture((64, 64), "rgba8Unorm", readable=False)
    with pytest.raises(ValueError, match="readable"):
        widget.present(tex)


def test_rejects_non_2d_texture(widget, device):
    tex = device.empty_texture((4, 64, 64), "rgba8Unorm")
    with pytest.raises(ValueError, match="2D"):
        widget.present(tex)


def test_rejects_other_device(widget):
    other = mtlpy.Device()
    with pytest.raises(ValueError, match="different Device"):
        widget.present(other.texture(_rgba(), "rgba8Unorm"))


@pytest.mark.parametrize("fmt", ["bgra8Unorm_srgb", "rgba16Float", "rgb10a2Unorm", "bgr10a2Unorm"])
def test_drawable_formats(app, device, fmt):
    w = MetalWidget(device, pixel_format=fmt)
    w.show()
    app.processEvents()
    try:
        assert w.present(device.texture(_rgba(), "rgba8Unorm"), wait=True)
    finally:
        _destroy(app, w)


def test_edr_options(app, device):
    w = MetalWidget(device, pixel_format="rgba16Float",
                    colorspace="extended_linear_srgb", edr=True,
                    scaling="fill", filter="nearest", display_sync=False,
                    max_drawables=2)
    w.show()
    app.processEvents()
    try:
        hdr = np.full((32, 32, 4), 2.0, np.float16)
        assert w.present(device.texture(hdr, "rgba16Float"), wait=True)
    finally:
        _destroy(app, w)


@pytest.mark.parametrize("kwargs,match", [
    ({"pixel_format": "rgba8Unorm"}, "pixel_format"),
    ({"scaling": "zoom"}, "scaling"),
    ({"colorspace": "adobe_rgb"}, "colorspace"),
    ({"filter": "cubic"}, "filter"),
    ({"max_drawables": 5}, "max_drawables"),
])
def test_invalid_options(app, device, kwargs, match):
    with pytest.raises(ValueError, match=match):
        MetalWidget(device, **kwargs)


def test_null_view(device):
    with pytest.raises(ValueError, match="non-null"):
        device.surface(0)


def test_close_is_idempotent(widget, device):
    tex = device.texture(_rgba(), "rgba8Unorm")
    widget.surface.close()
    widget.surface.close()
    assert widget.surface.closed
    assert widget.surface.drawable_size == (0, 0)
    with pytest.raises(RuntimeError, match="closed"):
        widget.present(tex)


def test_context_manager(app, device, widget):
    with device.surface(int(widget.winId())) as s:
        assert s.present(device.texture(_rgba(), "rgba8Unorm"), wait=True)
    assert s.closed


def test_surface_outlives_widget(app, device):
    # Qt destroys the NSView first; the Surface holds it weakly, so later
    # update_layout()/close()/GC must not touch freed memory.
    w = MetalWidget(device)
    w.show()
    app.processEvents()
    s = w.surface
    w.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del w
    s.update_layout()
    assert s.present(device.texture(_rgba(), "rgba8Unorm"), wait=True) in (True, False)
    s.close()


# -- zero-copy: next_frame() / present(frame) ---------------------------------

_FILL = """
#include <metal_stdlib>
using namespace metal;
kernel void fill(texture2d<float, access::write> t [[texture(0)]],
                 uint2 gid [[thread_position_in_grid]]) {
    t.write(float4(float(gid.x) / t.get_width(), 0.5, 0.0, 1.0), gid);
}
"""


@pytest.fixture
def fill(device):
    return device.compile(_FILL, "fill")


def test_next_frame_defaults_to_view_pixels(widget):
    frame = widget.next_frame()
    assert frame is not None
    assert (frame.width, frame.height) == widget.surface.view_size
    # Backing pixels, not points (2x on Retina).
    dpr = widget.devicePixelRatio()
    assert widget.surface.view_size == (round(widget.width() * dpr),
                                        round(widget.height() * dpr))
    widget.present(frame)


def test_zero_copy_kernel_writes_drawable(widget, fill):
    frame = widget.next_frame()
    tex = frame.texture
    assert tex is frame.texture  # one wrapper per frame
    assert tex.pixel_format == "bgra8Unorm" and tex.is_private
    assert (tex.width, tex.height) == (frame.width, frame.height)
    fill.run([], grid=(frame.width, frame.height, 1), textures=[tex], wait=False)
    assert widget.present(frame, wait=True)
    assert frame.presented


def test_zero_copy_loop_context_manager(widget, fill):
    for _ in range(30):
        frame = widget.next_frame()
        assert frame is not None
        with frame:
            fill.run([], grid=(frame.width, frame.height, 1),
                     textures=[frame.texture], wait=False)
        assert frame.presented


def test_context_manager_does_not_present_on_error(widget):
    frame = widget.next_frame()
    with pytest.raises(KeyError):
        with frame:
            raise KeyError
    assert not frame.presented


def test_next_frame_explicit_size(widget, fill):
    frame = widget.next_frame((1920, 1080))
    assert (frame.width, frame.height) == (1920, 1080)
    with frame:
        fill.run([], grid=(1920, 1080, 1), textures=[frame.texture], wait=False)
    assert widget.surface.drawable_size == (1920, 1080)


def test_dropped_frames_return_to_pool(widget):
    # Every un-presented Frame must give its drawable back on GC, or the
    # layer's (max_drawables=3) pool would run dry and next_frame() stall.
    for _ in range(10):
        assert widget.next_frame() is not None


def test_frame_present_twice(widget):
    frame = widget.next_frame()
    widget.present(frame)
    with pytest.raises(RuntimeError, match="already presented"):
        widget.present(frame)
    with pytest.raises(RuntimeError, match="already presented"):
        frame.texture


def test_frame_from_other_surface(app, device, widget):
    other = MetalWidget(device)
    other.show()
    app.processEvents()
    try:
        frame = other.next_frame()
        with pytest.raises(ValueError, match="different Surface"):
            widget.present(frame)
    finally:
        _destroy(app, other)


def test_zero_copy_waits_for_event(widget, device, fill):
    frame = widget.next_frame()
    producer, ev, result = _pending_producer(device)
    fill.run([], grid=(frame.width, frame.height, 1), textures=[frame.texture], cb=producer)
    producer.signal_event(ev, 1)
    _present_waiting_for_producer(widget, frame, producer, ev)
    assert result.contents[0] == 42
    assert frame.presented


def test_zero_copy_mixed_with_texture_present(widget, device, fill):
    frame = widget.next_frame()
    with frame:
        fill.run([], grid=(frame.width, frame.height, 1), textures=[frame.texture], wait=False)
    assert widget.present(device.texture(_rgba(), "rgba8Unorm"), wait=True)
    frame = widget.next_frame()
    assert (frame.width, frame.height) == widget.surface.view_size
    widget.present(frame)


def test_next_frame_tracks_resize(app, widget):
    for size in [(500, 300), (211, 97)]:
        widget.resize(*size)
        app.processEvents()
        dpr = widget.devicePixelRatio()
        expected = (round(size[0] * dpr), round(size[1] * dpr))
        assert widget.surface.view_size == expected
        frame = widget.next_frame()
        assert (frame.width, frame.height) == expected
        widget.present(frame)


def test_next_frame_hidden_returns_none(widget):
    widget.hide()
    assert widget.next_frame() is None


@pytest.mark.parametrize("fmt", ["bgra8Unorm_srgb", "rgba16Float"])
def test_zero_copy_formats(app, device, fill, fmt):
    w = MetalWidget(device, pixel_format=fmt)
    w.show()
    app.processEvents()
    try:
        frame = w.next_frame()
        assert frame.texture.pixel_format == fmt
        fill.run([], grid=(frame.width, frame.height, 1), textures=[frame.texture], wait=False)
        assert w.present(frame, wait=True)
    finally:
        _destroy(app, w)


def test_zero_copy_rejects_packed_format(app, device):
    w = MetalWidget(device, pixel_format="rgb10a2Unorm")
    try:
        with pytest.raises(ValueError, match="next_frame"):
            w.surface.next_frame()
    finally:
        _destroy(app, w)


def test_next_frame_bad_size(widget):
    with pytest.raises(ValueError, match="positive"):
        widget.next_frame((0, 100))


def test_last_present_gpu_time(widget, device, fill):
    big = device.empty_texture((2160, 3840), "rgba8Unorm")
    fill.run([], grid=(3840, 2160, 1), textures=[big])
    widget.present(big, wait=True)
    copy_time = widget.surface.last_present_gpu_time
    assert copy_time > 0  # the copy pass
    widget.present(widget.next_frame(), wait=True)
    # Relative, not an absolute threshold -- GPU timings vary run to run.
    assert widget.surface.last_present_gpu_time < copy_time


def test_zero_copy_frame_layout(widget, device):
    # A kernel writing RGBA red into the top-left quadrant must land in the
    # drawable as row 0 = top and BGRA byte order -- what Core Animation
    # composites. (The on-screen result is checked by eye with
    # examples/qt/orientation.py.)
    k = device.compile("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void top_left(texture2d<float, access::write> t [[texture(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
            bool tl = gid.x < t.get_width() / 2 && gid.y < t.get_height() / 2;
            t.write(tl ? float4(1, 0, 0, 1) : float4(0, 0, 0, 1), gid);
        }""", "top_left")
    frame = widget.next_frame((64, 32))
    k.run([], grid=(64, 32, 1), textures=[frame.texture])
    px = frame.texture.download()          # (H, W, 4) in BGRA memory order
    widget.present(frame)
    assert px.shape == (32, 64, 4)
    assert px[0, 0].tolist() == [0, 0, 255, 255]     # top-left: red, as B,G,R,A
    assert px[-1, -1].tolist() == [0, 0, 0, 255]     # bottom-right: black
    assert px[0, -1].tolist() == [0, 0, 0, 255]      # top-right: black
