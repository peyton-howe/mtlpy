#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>

namespace mtlpy {

class Buffer {
public:
    // storage_mode is a raw MTL::StorageMode value (Shared=0, Managed=1,
    // Private=2 -- see src/mtlpy/utils.py's StorageMode enum, which mirrors
    // these exactly so no translation is needed at the binding boundary).
    // Memoryless (3) isn't accepted: Metal restricts it to render-target
    // textures, not buffers.
    Buffer(MTL::Device* device, size_t size_bytes, uint32_t storage_mode);
    ~Buffer();

    // Only valid when storage_mode() == MTL::StorageModeShared -- throws
    // otherwise, since MTL::Buffer::contents() is nullptr for Private
    // storage, and a Managed buffer's CPU-side copy isn't guaranteed to be
    // in sync with the GPU's without an explicit blit-encoder
    // synchronizeResource() first. Device::copy_buffer() (blit-copy into a
    // fresh Shared buffer) is the storage-mode-agnostic way to read a
    // non-Shared Buffer's contents -- see Buffer.to_storage() in buffer.py,
    // which every CPU-reading path (contents/.numpy()/__dlpack__) goes
    // through first when needed.
    void*        contents_ptr() const;
    size_t       size_bytes()   const { return size_bytes_; }
    uint32_t     storage_mode() const { return storage_mode_; }
    MTL::Buffer* mtl()          const { return buf_; }

private:
    MTL::Buffer* buf_;
    size_t       size_bytes_;
    uint32_t     storage_mode_;
};

} // namespace mtlpy
