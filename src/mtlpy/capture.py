from __future__ import annotations


class Capture:
    """Returned by Device.start_capture() -- the capture has already started
    by the time this object exists; using it as a context manager is purely
    optional sugar for the matching stop_capture() call:

        with device.start_capture("trace.gputrace"):
            pipeline.run(bufs, grid)
        # stop_capture() already called here

    Calling device.stop_capture() directly (ignoring this object entirely)
    works exactly the same -- this doesn't hold any state of its own beyond
    a reference back to the Device that started the capture."""

    def __init__(self, device):
        self._device = device

    def __enter__(self) -> "Capture":
        return self

    def __exit__(self, *exc_info) -> None:
        self._device.stop_capture()


class CaptureScope:
    """A labeled begin/end marker (MTL::CaptureScope) that Xcode's GPU
    debugger -- or a .gputrace file opened later in Xcode -- shows as a
    named region in its capture timeline. See Device.capture_scope() and
    Device.start_capture(). Used as a context manager:

        with device.start_capture("trace.gputrace"):
            with device.capture_scope("Blur pass"):
                blur_pipeline.run(bufs, grid)
            with device.capture_scope("Sharpen pass"):
                sharpen_pipeline.run(bufs2, grid2)

    begin()/end() (or the context manager) are harmless no-ops if nothing is
    actually capturing right now (neither Device.start_capture() nor
    Xcode's own capture button/scheme setting) -- Metal only records scope
    boundaries while a capture is in progress."""

    def __init__(self, _scope, device):
        self._scope = _scope  # _mtlpy.CaptureScope
        self._device = device

    def begin(self) -> None:
        self._scope.begin_scope()

    def end(self) -> None:
        self._scope.end_scope()

    def __enter__(self) -> "CaptureScope":
        self.begin()
        return self

    def __exit__(self, *exc_info) -> None:
        self.end()
