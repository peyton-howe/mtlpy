#include "heap.h"
#include "buffer.h"
#include "texture.h"
#include "pool_guard.h"
#include <stdexcept>

namespace mtlpy {

Heap::Heap(MTL::Device* device, size_t size_bytes, uint32_t storage_mode)
    : storage_mode_(storage_mode)
{
    PoolGuard guard;
    // A Managed heap (unlike a standalone Managed Buffer, which Metal
    // quietly tolerates even where Managed isn't fully meaningful -- see
    // StorageMode's docstring) fails MTLHeapDescriptor's own device
    // validation on a unified-memory GPU (confirmed on an Apple silicon
    // Mac: this aborts the whole process via Objective-C's
    // NSError-to-abort path in
    // -[MTLHeapDescriptorInternal validateWithDevice:] -- not a catchable
    // Metal error, and not something supportsFamily(GPUFamilyMac2) can be
    // used to predict, since Apple silicon reports Mac2 support too).
    // hasUnifiedMemory() is the actual, confirmed gate -- check it
    // ourselves first so an unsupported request raises a normal,
    // catchable exception instead.
    if (storage_mode == MTL::StorageModeManaged && device->hasUnifiedMemory())
        throw std::runtime_error(
            "Device.heap(storage=StorageMode.MANAGED) is not supported on this GPU "
            "(Managed heaps aren't available on unified-memory GPUs, e.g. Apple "
            "silicon). Use StorageMode.SHARED or PRIVATE instead.");

    auto* desc = MTL::HeapDescriptor::alloc()->init();
    desc->setSize(size_bytes);
    desc->setStorageMode(static_cast<MTL::StorageMode>(storage_mode));

    heap_ = device->newHeap(desc);
    desc->release();

    if (!heap_)
        throw std::runtime_error("Failed to allocate Metal heap");
}

Heap::~Heap() {
    PoolGuard guard;
    heap_->release();
}

Buffer* Heap::new_buffer(size_t size_bytes) {
    PoolGuard guard;
    // Same MTL::ResourceOptions shift as Buffer's own device-allocating
    // constructor (see buffer.cpp) -- storage_mode_ is already a raw
    // MTL::StorageMode value.
    auto options = static_cast<MTL::ResourceOptions>(storage_mode_ << 4);
    auto* buf = heap_->newBuffer(size_bytes, options);
    if (!buf)
        throw std::runtime_error(
            "Heap.buffer()/.empty(): insufficient free space for a " +
            std::to_string(size_bytes) + "-byte buffer (heap size: " +
            std::to_string(heap_->size()) + " bytes, used: " +
            std::to_string(heap_->usedSize()) + " bytes)");
    return new Buffer(buf, size_bytes, storage_mode_);
}

Texture* Heap::new_texture(uint32_t dims, uint32_t pixel_format,
                            uint32_t width, uint32_t height, uint32_t depth,
                            uint32_t usage) {
    PoolGuard guard;
    MTL::TextureType type = texture_type_for(dims);

    auto* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(type);
    desc->setPixelFormat(static_cast<MTL::PixelFormat>(pixel_format));
    desc->setWidth(width);
    desc->setHeight(dims >= 2 ? height : 1);
    desc->setDepth(dims >= 3 ? depth : 1);
    desc->setStorageMode(static_cast<MTL::StorageMode>(storage_mode_));
    desc->setUsage(usage);

    auto* tex = heap_->newTexture(desc);
    desc->release();

    if (!tex)
        throw std::runtime_error(
            "Heap.empty_texture(): insufficient free space for this texture "
            "(heap size: " + std::to_string(heap_->size()) + " bytes, used: " +
            std::to_string(heap_->usedSize()) + " bytes)");
    bool is_private = storage_mode_ != MTL::StorageModeShared;
    return new Texture(tex, dims, is_private);
}

size_t Heap::size()      const { return (size_t)heap_->size(); }
size_t Heap::used_size() const { return (size_t)heap_->usedSize(); }

} // namespace mtlpy
