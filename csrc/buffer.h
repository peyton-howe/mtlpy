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

    // Wraps an already-allocated MTL::Buffer* (e.g. from MTL::Heap::newBuffer,
    // see Heap::new_buffer() in heap.cpp) -- takes ownership (the destructor
    // below releases it same as the device-allocating constructor above).
    // Callers are responsible for buf being non-null and storage_mode
    // matching its actual storage (Heap::new_buffer() derives both
    // correctly from the heap itself).
    Buffer(MTL::Buffer* buf, size_t size_bytes, uint32_t storage_mode);

    ~Buffer();

    // Only valid when storage_mode() == MTL::StorageModeShared -- throws
    // otherwise, since MTL::Buffer::contents() is nullptr for Private
    // storage, and a Managed buffer's CPU-side copy isn't guaranteed to be
    // in sync with the GPU's without an explicit blit-encoder
    // synchronizeResource() first. Device::copy_buffer() (blit-copy into a
    // fresh Shared buffer) is the storage-mode-agnostic way to read a
    // non-Shared Buffer's contents -- see Buffer.to_storage() in buffer.py,
    // which every public CPU-reading path (contents/.numpy()/__dlpack__)
    // goes through first, so this should never actually throw for a caller
    // going through the public Python API. It's a C++-level backstop for
    // anyone touching this class directly (a future native binding, or
    // Python code that reaches into the private `_buf` handle) -- as such
    // it throws a plain std::runtime_error (-> Python RuntimeError), not
    // the BufferError Buffer.__dlpack__ raises for the same underlying
    // condition on the public API: that's a deliberate, not accidental,
    // difference -- BufferError is the Python-idiomatic type for a
    // buffer-protocol-shaped failure (matches this project's existing
    // __dlpack__ convention), while RuntimeError is this codebase's
    // blanket convention for C++-thrown validation everywhere else (see
    // e.g. Texture::upload()/download()'s equivalent storage-mode guards).
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
