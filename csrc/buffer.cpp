#include "buffer.h"
#include "pool_guard.h"
#include <stdexcept>

namespace mtlpy {

Buffer::Buffer(MTL::Device* device, size_t size_bytes)
    : size_bytes_(size_bytes)
{
    PoolGuard guard;
    buf_ = device->newBuffer(size_bytes, MTL::ResourceStorageModeShared);
    if (!buf_)
        throw std::runtime_error("Failed to allocate Metal buffer");
}

Buffer::~Buffer() {
    PoolGuard guard;
    buf_->release();
}

void* Buffer::contents_ptr() const {
    return buf_->contents();
}

} // namespace mtlpy
