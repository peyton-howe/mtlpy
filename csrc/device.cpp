#include "device.h"
#include "buffer.h"
#include "command_buffer.h"
#include "metal_error.h"
#include "pipeline.h"
#include "pipeline_cache.h"
#include "sampler.h"
#include "texture.h"
#include <functional>
#include <stdexcept>

namespace mtlpy {

namespace {

// Shared by every blit-only Device method below (blit_upload_texture,
// optimize_texture_for_gpu_access, copy_texture): encode takes the one
// call as a lambda, error_context labels which operation failed if it does.
void run_blit(MTL::CommandQueue* queue, bool wait,
              const std::function<void(MTL::BlitCommandEncoder*)>& encode,
              const char* error_context) {
    auto* cmd = queue->commandBuffer();
    if (!cmd)
        throw std::runtime_error("Failed to create Metal command buffer");

    auto* blit = cmd->blitCommandEncoder();
    if (!blit)
        throw std::runtime_error("Failed to create Metal blit command encoder");

    encode(blit);
    blit->endEncoding();
    cmd->commit();

    if (wait) {
        cmd->waitUntilCompleted();
        throw_if_command_buffer_error(cmd, error_context);
    }
}

} // namespace

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
                                 uint32_t width, uint32_t height, uint32_t depth,
                                 uint32_t usage, bool private_storage) {
    return new Texture(device_, dims, pixel_format, width, height, depth, usage, private_storage);
}

void Device::blit_upload_texture(Buffer* buf, size_t offset, Texture* tex,
                                  size_t bytes_per_row, size_t bytes_per_image, bool wait) {
    MTL::Size size = MTL::Size::Make(
        tex->width(),
        tex->dims() >= 2 ? tex->height() : 1,
        tex->dims() >= 3 ? tex->depth()  : 1
    );
    run_blit(queue_, wait, [&](MTL::BlitCommandEncoder* blit) {
        blit->copyFromBuffer(buf->mtl(), offset, bytes_per_row, bytes_per_image, size,
                              tex->mtl(), /*destinationSlice=*/0, /*destinationLevel=*/0,
                              MTL::Origin(0, 0, 0));
    }, "GPU blit upload");
}

void Device::optimize_texture_for_gpu_access(Texture* tex, bool wait) {
    run_blit(queue_, wait, [&](MTL::BlitCommandEncoder* blit) {
        blit->optimizeContentsForGPUAccess(tex->mtl());
    }, "GPU texture optimization");
}

void Device::copy_texture(Texture* src, Texture* dst, bool wait) {
    run_blit(queue_, wait, [&](MTL::BlitCommandEncoder* blit) {
        blit->copyFromTexture(src->mtl(), dst->mtl());
    }, "GPU texture-to-texture copy");
}

Sampler* Device::create_sampler(bool linear, bool repeat) {
    return new Sampler(device_, linear, repeat);
}

CommandBuffer* Device::create_command_buffer() {
    return new CommandBuffer(queue_);
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
