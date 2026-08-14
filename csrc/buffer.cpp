#include "buffer.h"
#include "pool_guard.h"
#include <stdexcept>

namespace mtlpy {

Buffer::Buffer(MTL::Device* device, size_t size_bytes, uint32_t storage_mode)
    : size_bytes_(size_bytes), storage_mode_(storage_mode)
{
    PoolGuard guard;
    // MTL::ResourceStorageMode{Shared,Managed,Private} are exactly
    // MTL::StorageMode{Shared,Managed,Private} << 4 (see metal-cpp's
    // Metal/MTLResource.hpp) -- storage_mode is already a raw MTL::StorageMode
    // value, so this shift is the entire conversion.
    auto options = static_cast<MTL::ResourceOptions>(storage_mode << 4);
    buf_ = device->newBuffer(size_bytes, options);
    if (!buf_)
        throw std::runtime_error("Failed to allocate Metal buffer");
}

Buffer::Buffer(MTL::Buffer* buf, size_t size_bytes, uint32_t storage_mode)
    : buf_(buf), size_bytes_(size_bytes), storage_mode_(storage_mode)
{
}

Buffer::~Buffer() {
    PoolGuard guard;
    buf_->release();
}

void* Buffer::contents_ptr() const {
    if (storage_mode_ != MTL::StorageModeShared)
        throw std::runtime_error(
            "Buffer.contents/.numpy()/.data_ptr require Shared storage -- this "
            "Buffer is Private or Managed. Call Buffer.to_storage(StorageMode.SHARED) "
            "first (Buffer.numpy()/.contents already do this automatically).");
    return buf_->contents();
}

} // namespace mtlpy
