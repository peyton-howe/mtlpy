#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace mtlpy {

class Buffer;
class Pipeline;
class PipelineCache;
class Texture;
class Sampler;

class Device {
public:
    // index < 0 (the default) uses CreateSystemDefaultDevice(); index >= 0
    // selects that position in available_device_names()/CopyAllDevices(),
    // for multi-GPU machines.
    explicit Device(int index = -1);
    ~Device();

    Buffer*   create_buffer(size_t size_bytes);
    Pipeline* compile(const std::string& source, const std::string& function_name);

    // dims is 1/2/3 (see Texture); pixel_format is a raw MTL::PixelFormat
    // value (see src/mtlpy/utils.py's pixel format table). usage/
    // private_storage -- see Texture's constructor in texture.h.
    Texture* create_texture(uint32_t dims, uint32_t pixel_format,
                             uint32_t width, uint32_t height, uint32_t depth,
                             uint32_t usage, bool private_storage);

    // Hardware-blit upload: copies buf's memory into tex via
    // MTLBlitCommandEncoder rather than Texture::upload()'s CPU-side
    // replaceRegion. tex keeps its normal (possibly tiled/swizzled)
    // internal layout -- the blit engine does the retiling on the GPU side,
    // concurrently with the CPU, instead of the CPU computing it inline.
    // bytes_per_row/bytes_per_image describe buf's layout starting at
    // offset (src/mtlpy/texture.py computes these as tightly packed, same
    // convention as Texture::upload's bytes_per_row).
    void blit_upload_texture(Buffer* buf, size_t offset, Texture* tex,
                              size_t bytes_per_row, size_t bytes_per_image, bool wait);

    // Encodes MTLBlitCommandEncoder::optimizeContentsForGPUAccess -- lets
    // Metal repack a texture's contents into its preferred GPU-side layout
    // after the fact. Private-storage textures already get this for free at
    // creation (per Apple's docs); this exists for the Shared-storage case,
    // which doesn't. tex's contents must already be populated (upload()/
    // upload_from_buffer()) before calling this.
    void optimize_texture_for_gpu_access(Texture* tex, bool wait);

    // Hardware-blit texture-to-texture copy (MTLBlitCommandEncoder::
    // copyFromTexture, whole-texture overload): src and dst must already
    // match in pixel format and dimensions -- this moves raw bytes, no
    // shader/format-conversion path, so it works for any pixel format
    // (including Unorm, unlike Texture::to_buffer()) and any combination of
    // Shared/Private storage on either side.
    void copy_texture(Texture* src, Texture* dst, bool wait);

    Sampler* create_sampler(bool linear, bool repeat);

    uint32_t max_threads_per_threadgroup() const;
    void     flush_cache();

    static std::vector<std::string> available_device_names();

private:
    MTL::Device*       device_;
    MTL::CommandQueue* queue_;
    PipelineCache*     cache_;
};

} // namespace mtlpy
