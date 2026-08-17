#include "fence.h"
#include "pool_guard.h"
#include <stdexcept>

namespace mtlpy {

Fence::Fence(MTL::Device* device) {
    PoolGuard guard;
    fence_ = device->newFence();
    if (!fence_)
        throw std::runtime_error("Failed to create Metal fence");
}

Fence::~Fence() {
    PoolGuard guard;
    fence_->release();
}

} // namespace mtlpy
