"""Correctness for Device.start_capture()/.stop_capture()/.capture_scope() --
Metal's programmatic GPU frame capture (MTLCaptureManager), for the Xcode GPU
debugger or a .gputrace file.

Actually capturing successfully requires the MTL_CAPTURE_ENABLED=1
environment variable to be present when the Metal framework first
initializes in this process (Metal conditionally loads its capture-layer
interposer at that point) -- confirmed by hand that setting it later,
even before creating a *fresh* Device, is too late once the framework has
already initialized without it in this process. That's not something a
pytest test body can retroactively arrange (this whole test process is
already running by the time any test executes), so this file only exercises
what's reliable in an ordinary test environment: the clean-failure paths, and
that capture_scope()'s begin/end are harmless no-ops with no capture active.
The success path (capture to a .gputrace file, verified to actually produce
a valid trace bundle) was confirmed by hand with
`MTL_CAPTURE_ENABLED=1 python3 -c ...` -- see this PR's description.
"""
import numpy as np
import pytest

try:
    from mtlpy import shader
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_is_capturing_false_by_default(device):
    assert device.is_capturing is False


def test_capture_scope_begin_end_is_harmless_without_active_capture(device):
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(3, dtype=np.float32))
    b = device.buffer(np.ones(3, dtype=np.float32))
    out = device.empty(3, np.float32)

    scope = device.capture_scope("smoke test scope")
    scope.begin()
    add_pipeline.run([a, b, out], 3)
    scope.end()

    np.testing.assert_allclose(out.contents, [2.0, 2.0, 2.0])


def test_capture_scope_as_context_manager(device):
    add_pipeline = device.compile(shader.add_kernel("float"), "add")
    a = device.buffer(np.ones(3, dtype=np.float32))
    b = device.buffer(np.ones(3, dtype=np.float32))
    out = device.empty(3, np.float32)

    with device.capture_scope("ctx manager scope"):
        add_pipeline.run([a, b, out], 3)

    np.testing.assert_allclose(out.contents, [2.0, 2.0, 2.0])


def test_capture_scope_with_no_label(device):
    # label is optional -- must not require one.
    with device.capture_scope():
        pass


def test_start_capture_without_xcode_or_path_raises(device):
    # No Xcode GPU debugger is attached in a normal test run, and no path=
    # is given here -- Metal has nowhere to send the capture, so this must
    # raise a catchable RuntimeError rather than crash or hang.
    with pytest.raises(RuntimeError):
        device.start_capture()
    # Must not have left a capture dangling after the failed attempt.
    assert device.is_capturing is False


def test_start_capture_to_file_without_capture_enabled_raises(tmp_path, device):
    # Even with a path= (so destination isn't the missing-Xcode problem
    # above), MTL_CAPTURE_ENABLED=1 isn't set for this test process, so
    # Metal's capture layer was never inserted -- must raise cleanly rather
    # than crash.
    with pytest.raises(RuntimeError):
        device.start_capture(str(tmp_path / "trace.gputrace"))
    assert device.is_capturing is False
