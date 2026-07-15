"""Texture and Sampler: creation, upload/download roundtrips, a read-write
compute kernel, and a sampling kernel. Unlike Buffer, Texture has no
zero-copy .contents -- see Texture's docstring in src/mtlpy/texture.py for
why (Metal doesn't guarantee a texture's CPU-visible layout is a tightly
packed array the way a Shared-storage Buffer's is).
"""
import numpy as np
import pytest

try:
    from mtlpy import shader
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_shape_single_channel_2d(device):
    tex = device.empty_texture((4, 6), "r32Float")  # (height, width)
    assert tex.shape == (4, 6)
    assert tex.dims == 2


def test_shape_multi_channel_2d(device):
    tex = device.empty_texture((4, 6), "rgba8Unorm")
    assert tex.shape == (4, 6, 4)


def test_shape_1d_and_3d(device):
    assert device.empty_texture((10,), "r32Float").shape == (10,)
    assert device.empty_texture((2, 3, 4), "r32Float").shape == (2, 3, 4)


def test_unknown_pixel_format_raises(device):
    with pytest.raises(ValueError):
        device.empty_texture((4, 4), "not_a_real_format")


@pytest.mark.parametrize("pixel_format", ["r32Float", "rgba32Float", "r8Unorm", "rgba8Unorm"])
def test_upload_download_roundtrip_2d(device, pixel_format):
    height, width = 5, 7
    tex = device.empty_texture((height, width), pixel_format)
    rng = np.random.default_rng(0)
    if tex.dtype == np.uint8:
        data = rng.integers(0, 256, size=tex.shape, dtype=np.uint8)
    else:
        data = rng.random(tex.shape, dtype=np.float32).astype(tex.dtype)

    tex.upload(data)
    np.testing.assert_array_equal(tex.download(), data)
    np.testing.assert_array_equal(np.asarray(tex), data)  # __array__


def test_upload_download_roundtrip_1d(device):
    # Regression test: Metal requires bytes_per_row/bytes_per_image to be 0
    # for a 1D texture's replaceRegion/getBytes calls -- this path was
    # previously untested and passed a non-zero bytes_per_row.
    tex = device.empty_texture((16,), "r32Float")
    data = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    tex.upload(data)
    np.testing.assert_allclose(tex.download(), data, atol=1e-6)


def test_empty_texture_3d_width_matching_channel_count_is_not_rejected(device):
    # (8, 8, 4) for a 4-channel format is ambiguous from shape alone: it
    # could be an (H, W, C) array someone forgot to strip the channel axis
    # from, or a genuine depth=8/height=8/width=4 3D texture request.
    tex = device.empty_texture((8, 8, 4), "rgba8Unorm")
    assert tex.dims == 3
    assert (tex.depth, tex.height, tex.width) == (8, 8, 4)
    # .shape reports the *spatial* shape (8, 8, 4) plus rgba8Unorm's own
    # trailing channel axis, per Texture.shape's documented convention.
    assert tex.shape == (8, 8, 4, 4)


def test_device_texture_from_ndarray(device):
    img = np.arange(4 * 6 * 4, dtype=np.uint8).reshape(4, 6, 4)
    tex = device.texture(img, "rgba8Unorm")
    assert tex.shape == (4, 6, 4)
    np.testing.assert_array_equal(tex.download(), img)


def test_upload_rejects_wrong_shape(device):
    tex = device.empty_texture((4, 6), "r32Float")
    with pytest.raises(ValueError):
        tex.upload(np.zeros((5, 6), dtype=np.float32))


_INVERT_2D_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void invert(
    texture2d<float, access::read_write> tex [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float4 c = tex.read(gid);
    tex.write(1.0 - c, gid);
}
"""


def test_readwrite_texture_kernel(device):
    height, width = 4, 4
    tex = device.empty_texture((height, width), "r32Float")
    data = np.full((height, width), 0.25, dtype=np.float32)
    tex.upload(data)

    pipeline = device.compile(_INVERT_2D_SOURCE, "invert")
    pipeline.run([], grid=(width, height, 1), textures=[tex])

    np.testing.assert_allclose(tex.download(), np.full((height, width), 0.75), atol=1e-6)


_SAMPLE_COPY_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void sample_copy(
    texture2d<float, access::sample>     src [[texture(0)]],
    texture2d<float, access::write>      dst [[texture(1)]],
    sampler                              smp [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    dst.write(src.sample(smp, uv), gid);
}
"""


def test_sampling_kernel(device):
    height, width = 4, 4
    src = device.empty_texture((height, width), "r32Float")
    dst = device.empty_texture((height, width), "r32Float")
    data = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
    src.upload(data)

    smp = device.sampler(linear=False)  # nearest -- exact passthrough at texel centers
    pipeline = device.compile(_SAMPLE_COPY_SOURCE, "sample_copy")
    pipeline.run([], grid=(width, height, 1), textures=[src, dst], samplers=[smp])

    np.testing.assert_allclose(dst.download(), data, atol=1e-6)


def test_run_with_too_few_textures_raises(device):
    # Mirrors test_basic.py::test_run_with_too_few_buffers_raises -- an
    # unbound texture argument is undefined behavior in Metal, not a safe
    # no-op, so Pipeline.run must catch it itself via reflection.
    pipeline = device.compile(_INVERT_2D_SOURCE, "invert")
    with pytest.raises(RuntimeError):
        pipeline.run([], grid=(4, 4, 1), textures=[])  # missing tex


def test_run_with_too_few_samplers_raises(device):
    height, width = 4, 4
    src = device.empty_texture((height, width), "r32Float")
    dst = device.empty_texture((height, width), "r32Float")
    pipeline = device.compile(_SAMPLE_COPY_SOURCE, "sample_copy")
    with pytest.raises(RuntimeError):
        pipeline.run([], grid=(width, height, 1), textures=[src, dst], samplers=[])  # missing smp


def test_texture_type_helper():
    assert shader.texture_type(2, "float") == "texture2d<float, access::read_write>"
    assert shader.texture_type(1, "half", "read") == "texture1d<half, access::read>"
    with pytest.raises(ValueError):
        shader.texture_type(4, "float")
