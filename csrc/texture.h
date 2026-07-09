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
    Texture(MTL::Device* device, uint32_t dims, uint32_t pixel_format,
            uint32_t width, uint32_t height, uint32_t depth);
    ~Texture();

    // Metal textures don't expose a directly-addressable CPU pointer the
    // way a Shared-storage-mode Buffer does (internal row layout/padding
    // can differ from a tightly packed array) -- replaceRegion/getBytes,
    // a genuine copy, is the only sanctioned way to move data in or out.
    void upload(const void* bytes, size_t bytes_per_row, size_t bytes_per_image);
    void download(void* bytes, size_t bytes_per_row, size_t bytes_per_image) const;

    uint32_t width()  const;
    uint32_t height() const;
    uint32_t depth()  const;
    uint32_t dims()   const { return dims_; }

    MTL::Texture* mtl() const { return tex_; }

private:
    MTL::Texture* tex_;
    uint32_t      dims_;
};

} // namespace mtlpy
