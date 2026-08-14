#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>

namespace mtlpy {

class Buffer;
class Texture;

// A memory pool (MTL::Heap) that Buffers/Textures are sub-allocated from
// instead of each getting its own standalone MTL::Device allocation.
// Every resource sub-allocated from a Heap shares the Heap's own storage
// mode -- Metal's own constraint (set once here, not accepted per-resource
// by new_buffer()/new_texture() below).
//
// This is the minimal Metal heap surface: automatic-type heaps only (no
// placement/sparse heaps), no aliasing control (every resource is
// non-aliased, Metal's default), no purgeability API.
class Heap {
public:
    // storage_mode -- raw MTL::StorageMode value, same convention as
    // Buffer's constructor (see buffer.h).
    Heap(MTL::Device* device, size_t size_bytes, uint32_t storage_mode);
    ~Heap();

    Buffer* new_buffer(size_t size_bytes);

    // dims/pixel_format/width/height/depth/usage -- see Texture's own
    // constructor in texture.h. This Heap's storage mode applies
    // regardless of what a bare Device::create_texture() private_storage
    // bool would otherwise pick.
    Texture* new_texture(uint32_t dims, uint32_t pixel_format,
                          uint32_t width, uint32_t height, uint32_t depth,
                          uint32_t usage);

    size_t   size()         const;
    size_t   used_size()    const;
    uint32_t storage_mode() const { return storage_mode_; }
    MTL::Heap* mtl()        const { return heap_; }

private:
    MTL::Heap* heap_;
    uint32_t   storage_mode_;
};

} // namespace mtlpy
