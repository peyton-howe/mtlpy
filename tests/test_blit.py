"""Public blit-copy API on Device (see src/mtlpy/device.py): Device.copy_buffer()
(buffer-to-buffer), Device.blit_upload_texture()/blit_download_texture()
(buffer<->texture) -- all hardware MTLBlitCommandEncoder copies, exposed
directly instead of only reachable through Buffer.to_storage()/
Texture.upload_from_buffer(), which now delegate to these same methods.
"""
import numpy as np
import pytest

try:
    import mtlpy
    from mtlpy import Device, StorageMode
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


# ---------------------------------------------------------------------------
# Device.copy_buffer()
# ---------------------------------------------------------------------------

def test_copy_buffer_full(device):
    src = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    dst = device.empty(4, np.float32)
    device.copy_buffer(src, dst)
    np.testing.assert_array_equal(dst.numpy(), [1.0, 2.0, 3.0, 4.0])


def test_copy_buffer_default_size_bytes_copies_rest_of_src(device):
    src = device.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    dst = device.empty(4, np.float32)
    itemsize = np.dtype(np.float32).itemsize
    device.copy_buffer(src, dst, src_offset=itemsize)  # skip the first element
    np.testing.assert_array_equal(dst.numpy()[:3], [2.0, 3.0, 4.0])


def test_copy_buffer_with_offsets(device):
    src = device.buffer(np.array([10.0, 20.0], dtype=np.float32))
    dst = device.buffer(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    itemsize = np.dtype(np.float32).itemsize
    device.copy_buffer(src, dst, dst_offset=2 * itemsize, size_bytes=2 * itemsize)
    np.testing.assert_array_equal(dst.numpy(), [0.0, 0.0, 10.0, 20.0])


def test_copy_buffer_works_across_storage_modes(device):
    src = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32), storage=StorageMode.PRIVATE)
    dst = device.empty(3, np.float32, storage=StorageMode.PRIVATE)
    device.copy_buffer(src, dst)
    np.testing.assert_array_equal(dst.numpy(), [1.0, 2.0, 3.0])


def test_copy_buffer_cross_device_raises():
    dev1, dev2 = Device(), Device()
    src = dev1.buffer(np.array([1.0], dtype=np.float32))
    dst = dev2.empty(1, np.float32)
    with pytest.raises(ValueError):
        dev1.copy_buffer(src, dst)


def test_copy_buffer_out_of_range_raises(device):
    src = device.empty(4, np.float32)
    dst = device.empty(4, np.float32)
    with pytest.raises(RuntimeError):
        device.copy_buffer(src, dst, size_bytes=4 * np.dtype(np.float32).itemsize + 1)


def test_buffer_to_storage_still_works_after_refactor(device):
    """to_storage() now delegates to the public Device.copy_buffer() instead
    of reaching into _dev.copy_buffer() directly -- pin down the round trip
    still holds."""
    buf = device.buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    private = buf.to_storage(StorageMode.PRIVATE)
    back = private.to_storage(StorageMode.SHARED)
    np.testing.assert_array_equal(back.numpy(), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Device.blit_upload_texture()
# ---------------------------------------------------------------------------

def test_blit_upload_texture_direct(device):
    tex = device.empty_texture((2, 2), "r32Float")
    staging = device.buffer(np.arange(4, dtype=np.float32))
    device.blit_upload_texture(staging, tex)
    np.testing.assert_array_equal(tex.numpy().reshape(-1), np.arange(4, dtype=np.float32))


def test_texture_upload_from_buffer_matches_direct_device_call(device):
    """Texture.upload_from_buffer() now delegates to Device.blit_upload_texture()
    -- confirm both paths produce the same result."""
    data = np.arange(4, dtype=np.float32)
    tex_a = device.empty_texture((2, 2), "r32Float")
    tex_a.upload_from_buffer(device.buffer(data))
    tex_b = device.empty_texture((2, 2), "r32Float")
    device.blit_upload_texture(device.buffer(data), tex_b)
    np.testing.assert_array_equal(tex_a.numpy(), tex_b.numpy())


def test_blit_upload_texture_dtype_mismatch_raises(device):
    tex = device.empty_texture((2, 2), "r32Float")
    wrong_dtype_buf = device.buffer(np.arange(4, dtype=np.int32))
    with pytest.raises(TypeError):
        device.blit_upload_texture(wrong_dtype_buf, tex)


def test_blit_upload_texture_cross_device_raises():
    dev1, dev2 = Device(), Device()
    tex = dev1.empty_texture((2, 2), "r32Float")
    buf = dev2.buffer(np.arange(4, dtype=np.float32))
    with pytest.raises(ValueError):
        dev1.blit_upload_texture(buf, tex)


# ---------------------------------------------------------------------------
# Device.blit_download_texture() / Texture.download_to_buffer()
# ---------------------------------------------------------------------------

def test_blit_download_texture_roundtrip(device):
    tex = device.empty_texture((2, 2), "r32Float")
    data = np.arange(4, dtype=np.float32)
    tex.upload(data.reshape(2, 2))
    out = device.empty(4, np.float32)
    device.blit_download_texture(tex, out)
    np.testing.assert_array_equal(out.numpy(), data)


def test_download_to_buffer_roundtrip(device):
    tex = device.empty_texture((2, 2), "r32Float")
    data = np.arange(4, dtype=np.float32)
    tex.upload(data.reshape(2, 2))
    out = device.empty(4, np.float32)
    tex.download_to_buffer(out)
    np.testing.assert_array_equal(out.numpy(), data)


def test_download_to_buffer_with_offset(device):
    tex = device.empty_texture((1, 2), "r32Float")
    tex.upload(np.array([[5.0, 6.0]], dtype=np.float32))
    out = device.buffer(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    itemsize = np.dtype(np.float32).itemsize
    tex.download_to_buffer(out, offset=2 * itemsize)
    np.testing.assert_array_equal(out.numpy(), [0.0, 0.0, 5.0, 6.0])


def test_blit_download_texture_works_on_unorm(device):
    """A blit copy moves raw bytes with no shader/format-conversion pass,
    so it works for Unorm formats the same as any other pixel format."""
    tex = device.empty_texture((2, 2), "rgba8Unorm")
    data = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
    tex.upload(data)
    out = device.empty(16, np.uint8)
    tex.download_to_buffer(out)
    np.testing.assert_array_equal(out.numpy().reshape(2, 2, 4), data)


def test_blit_download_texture_works_on_unreadable_texture(device):
    """buffer_from_texture()/.to_buffer() require readable=True (their
    compute kernel does texture.read()) -- a blit copy needs no shader
    access at all, so this must work where those would raise ValueError."""
    tex = device.empty_texture((2, 2), "r32Float", readable=False, writable=True)
    data = np.arange(4, dtype=np.float32)
    tex.upload(data.reshape(2, 2))
    with pytest.raises(ValueError):
        tex.to_buffer()
    out = device.empty(4, np.float32)
    tex.download_to_buffer(out)
    np.testing.assert_array_equal(out.numpy(), data)


def test_blit_download_texture_works_on_private_texture(device):
    tex = device.empty_texture((2, 2), "r32Float", private=True)
    staging_in = device.buffer(np.arange(4, dtype=np.float32))
    tex.upload_from_buffer(staging_in)
    out = device.empty(4, np.float32)
    tex.download_to_buffer(out)
    np.testing.assert_array_equal(out.numpy(), np.arange(4, dtype=np.float32))


def test_download_works_on_private_texture(device):
    """Texture.download() is blit-based internally (see its docstring)
    instead of getBytes()-based, so it works on a private=True texture."""
    tex = device.empty_texture((2, 2), "r32Float", private=True)
    staging_in = device.buffer(np.arange(4, dtype=np.float32))
    tex.upload_from_buffer(staging_in)
    np.testing.assert_array_equal(tex.download(), np.arange(4, dtype=np.float32).reshape(2, 2))
    np.testing.assert_array_equal(tex.numpy(), np.arange(4, dtype=np.float32).reshape(2, 2))


def test_upload_works_on_private_texture(device):
    """Texture.upload() is also blit-based internally now (see its
    docstring) instead of replaceRegion-based, so it too works on a
    private=True texture -- it no longer raises like it used to."""
    tex = device.empty_texture((2, 2), "r32Float", private=True)
    tex.upload(np.arange(4, dtype=np.float32).reshape(2, 2))
    np.testing.assert_array_equal(tex.download(), np.arange(4, dtype=np.float32).reshape(2, 2))


def test_blit_download_texture_dtype_mismatch_raises(device):
    tex = device.empty_texture((2, 2), "r32Float")
    wrong_dtype_buf = device.empty(4, np.int32)
    with pytest.raises(TypeError):
        device.blit_download_texture(tex, wrong_dtype_buf)


def test_blit_download_texture_cross_device_raises():
    dev1, dev2 = Device(), Device()
    tex = dev1.empty_texture((2, 2), "r32Float")
    buf = dev2.empty(4, np.float32)
    with pytest.raises(ValueError):
        dev1.blit_download_texture(tex, buf)


# ---------------------------------------------------------------------------
# Device._staging_buffer() -- the internal Heap backing Texture.upload()/
# .download() (see their docstrings and Device._staging_buffer()'s own).
# ---------------------------------------------------------------------------

def test_download_holding_many_results_alive_does_not_exhaust_staging_heap(device):
    """The whole point of .download()'s eager copy-out: each call's
    staging Buffer is unreferenced (and its heap space reclaimed) by the
    time this returns, so holding many results alive simultaneously must
    never raise, unlike a naive zero-copy-view-into-a-shared-heap design
    would (see Heap's class docstring for that failure mode)."""
    tex = device.empty_texture((64, 64), "r32Float")
    tex.upload(np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    results = [tex.download() for _ in range(50)]  # held alive simultaneously
    for r in results:
        np.testing.assert_array_equal(r, results[0])


def test_download_result_is_independent_copy_not_a_shared_view(device):
    """Two .download() calls must not alias the same memory -- each must
    be a fully independent array (this is what the eager .copy() in
    .download() guarantees; a naive reused-buffer implementation would
    fail this)."""
    tex = device.empty_texture((2, 2), "r32Float")
    tex.upload(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    a = tex.download()
    tex.upload(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    b = tex.download()
    np.testing.assert_array_equal(a, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(b, [[5.0, 6.0], [7.0, 8.0]])


def test_staging_heap_grows_for_a_bigger_texture(device):
    """Device._staging_buffer()'s Heap starts small (or nonexistent) and
    must grow to fit a bigger request later, rather than failing."""
    small = device.empty_texture((2, 2), "r32Float")
    small.upload(np.zeros((2, 2), dtype=np.float32))
    small.download()  # heap now sized for a 2x2 texture

    big = device.empty_texture((256, 256), "r32Float")
    data = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    big.upload(data)  # must grow the staging heap, not fail
    np.testing.assert_array_equal(big.download(), data)


def test_staging_heap_shared_across_textures_on_same_device(device):
    """Device._staging_buffer() is a Device-level Heap, not a per-Texture
    one -- confirm two different Textures on the same Device reuse it."""
    tex1 = device.empty_texture((4, 4), "r32Float")
    tex1.upload(np.zeros((4, 4), dtype=np.float32))
    tex1.download()
    heap_after_first = device._staging_heap
    assert heap_after_first is not None

    tex2 = device.empty_texture((4, 4), "r32Float")
    tex2.upload(np.ones((4, 4), dtype=np.float32))
    tex2.download()
    assert device._staging_heap is heap_after_first  # same heap, not a new one


def test_clear_staging_heap_before_any_use_returns_zero(device):
    assert device.clear_staging_heap() == 0
    assert device._staging_heap is None


def test_clear_staging_heap_frees_and_reports_size(device):
    tex = device.empty_texture((256, 256), "r32Float")
    tex.upload(np.zeros((256, 256), dtype=np.float32))
    tex.download()
    heap_size = device._staging_heap.size
    assert heap_size > 0

    freed = device.clear_staging_heap()
    assert freed == heap_size
    assert device._staging_heap is None


def test_upload_download_work_normally_after_clearing_staging_heap(device):
    """clear_staging_heap() must not break subsequent calls -- the next
    .upload()/.download() lazily recreates the Heap."""
    tex = device.empty_texture((256, 256), "r32Float")
    tex.upload(np.zeros((256, 256), dtype=np.float32))
    tex.download()
    device.clear_staging_heap()

    data = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    tex.upload(data)
    np.testing.assert_array_equal(tex.download(), data)
    assert device._staging_heap is not None
    # Recreated small (sized for this request), not still at the old high-water mark
    assert device._staging_heap.size < 4 * 1024 * 1024
