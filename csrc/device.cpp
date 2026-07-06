#include "device.h"
#include "buffer.h"
#include "pipeline.h"
#include "pipeline_cache.h"
#include <stdexcept>

namespace mtlpy {

Device::Device() {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_)
        throw std::runtime_error("No Metal-capable device found");

    queue_ = device_->newCommandQueue();
    if (!queue_) {
        device_->release();
        throw std::runtime_error("Failed to create Metal command queue");
    }

    cache_ = new PipelineCache(device_);
}

Device::~Device() {
    delete cache_;
    queue_->release();
    device_->release();
}

Buffer* Device::create_buffer(size_t size_bytes) {
    return new Buffer(device_, size_bytes);
}

Pipeline* Device::compile(const std::string& source, const std::string& function_name) {
    auto cached = cache_->get_or_create(device_, source, function_name);
    return new Pipeline(cached.state, queue_, cached.required_buffer_count);
}

uint32_t Device::max_threads_per_threadgroup() const {
    return (uint32_t)device_->maxThreadsPerThreadgroup().width;
}

} // namespace mtlpy
