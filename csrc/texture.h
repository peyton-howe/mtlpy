#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>

namespace mtlpy {

class Texture {
public:
    // dims selects MTL::TextureType1D/2D/3D (1, 2, or 3 -- no arrays or
    // multisampling). pixel_format is a raw MTL::PixelFormat value (see
    // src/mtlpy/utils.py's pixel format table for the values this project
    // supports); height/depth are ignored (must be 1) below their dims.
    //
    // usage is a bitmask of MTL::TextureUsageShaderRead/ShaderWrite --
    // declare only what the kernel(s) actually need. private_storage picks
    // MTL::StorageModePrivate (GPU-only memory, letting Metal use its most
    // aggressive internal tiling/compression) over the default
    // StorageModeShared (CPU-visible, required by upload()/download()
    // below). Both matter for the same reason: a texture Metal knows the
    // CPU can never see and the GPU will never write can be laid out more
    // aggressively than one that must stay generically accessible -- see
    // Device::create_texture's docstring-equivalent in device.h for the
    // measured effect. A private_storage texture's CPU interop must go
    // through Device::blit_upload_texture()/buffer_from_texture() instead
    // of upload()/download(), which Metal rejects on Private storage.
    Texture(MTL::Device* device, uint32_t dims, uint32_t pixel_format,
            uint32_t width, uint32_t height, uint32_t depth,
            uint32_t usage, bool private_storage);
    ~Texture();

    // Metal textures don't expose a directly-addressable CPU pointer the
    // way a Shared-storage-mode Buffer does (internal row layout/padding
    // can differ from a tightly packed array) -- replaceRegion/getBytes,
    // a genuine copy, is the only sanctioned way to move data in or out.
    // Only valid when private_storage was false at construction (see above).
    void upload(const void* bytes, size_t bytes_per_row, size_t bytes_per_image);
    void download(void* bytes, size_t bytes_per_row, size_t bytes_per_image) const;

    uint32_t width()  const;
    uint32_t height() const;
    uint32_t depth()  const;
    uint32_t dims()   const { return dims_; }
    bool     is_private() const { return is_private_; }

    MTL::Texture* mtl() const { return tex_; }

private:
    MTL::Texture* tex_;
    uint32_t      dims_;
    bool          is_private_;
};

} // namespace mtlpy
