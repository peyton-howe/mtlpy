#include "device.h"
#include "buffer.h"
#include "pipeline.h"
#include "pipeline_cache.h"
#include "sampler.h"
#include "texture.h"
#include <stdexcept>

namespace mtlpy {

Device::Device(int index) {
    if (index < 0) {
        device_ = MTL::CreateSystemDefaultDevice();
    } else {
        // CopyAllDevices() returns +1-owned array whose elements it retains
        // only for its own lifetime -- explicitly retain the one we're
        // keeping before releasing the array, standard manual-refcounting
        // pattern for Cocoa "copy" APIs.
        auto* all = MTL::CopyAllDevices();
        if (!all || (NS::UInteger)index >= all->count()) {
            if (all)
                all->release();
            throw std::runtime_error(
                "GPU index " + std::to_string(index) + " out of range "
                "(see mtlpy.list_devices() for available GPUs)");
        }
        device_ = all->object<MTL::Device>((NS::UInteger)index);
        device_->retain();
        all->release();
    }
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
    return new Pipeline(cached.state, queue_, cached.required_buffer_count,
                         cached.required_texture_count, cached.required_sampler_count);
}

Texture* Device::create_texture(uint32_t dims, uint32_t pixel_format,
                                 uint32_t width, uint32_t height, uint32_t depth) {
    return new Texture(device_, dims, pixel_format, width, height, depth);
}

Sampler* Device::create_sampler(bool linear, bool repeat) {
    return new Sampler(device_, linear, repeat);
}

uint32_t Device::max_threads_per_threadgroup() const {
    return (uint32_t)device_->maxThreadsPerThreadgroup().width;
}

void Device::flush_cache() {
    cache_->flush();
}

std::vector<std::string> Device::available_device_names() {
    std::vector<std::string> names;
    auto* all = MTL::CopyAllDevices();
    if (all) {
        for (NS::UInteger i = 0; i < all->count(); ++i)
            names.push_back(all->object<MTL::Device>(i)->name()->utf8String());
        all->release();
    }
    return names;
}

} // namespace mtlpy
