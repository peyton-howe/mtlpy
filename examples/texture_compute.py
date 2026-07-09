"""Create a texture, run a read-write compute kernel over it, and a
sampling kernel that reads it into a second texture.

Unlike Buffer, Texture has no zero-copy .contents -- .upload()/.download()
are genuine copies (see src/mtlpy/texture.py for why).
"""
import numpy as np
import mtlpy

device = mtlpy.Device()

height, width = 8, 8
checkerboard = (np.indices((height, width)).sum(axis=0) % 2).astype(np.float32)

tex = device.texture(checkerboard, "r32Float")
print("uploaded:\n", tex.download())

invert_source = """
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
invert = device.compile(invert_source, "invert")
invert.run([], grid=(width, height, 1), textures=[tex])
print("inverted:\n", tex.download())

# Sample tex (bilinear this time) into a smaller texture -- a cheap downscale.
sample_source = """
#include <metal_stdlib>
using namespace metal;
kernel void downscale(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    sampler                          smp [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    dst.write(src.sample(smp, uv), gid);
}
"""
small = device.empty_texture((height // 2, width // 2), "r32Float")
linear_sampler = device.sampler(linear=True)
downscale = device.compile(sample_source, "downscale")
downscale.run([], grid=(width // 2, height // 2, 1), textures=[tex, small], samplers=[linear_sampler])
print("downscaled:\n", small.download())
