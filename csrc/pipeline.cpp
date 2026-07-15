#include "pipeline.h"
#include "buffer.h"
#include "sampler.h"
#include "texture.h"
#include <Foundation/NSAutoreleasePool.hpp>
#include <cmath>
#include <stdexcept>

namespace mtlpy {

namespace {
// RAII wrapper so the pool is always drained, including on the exception
// paths in Pipeline::run().
struct PoolGuard {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    ~PoolGuard() { pool->release(); }
};
} // namespace

Pipeline::Pipeline(MTL::ComputePipelineState* state, MTL::CommandQueue* queue,
                    uint32_t required_buffer_count, uint32_t required_texture_count,
                    uint32_t required_sampler_count)
    : state_(state), queue_(queue),
      required_buffer_count_(required_buffer_count),
      required_texture_count_(required_texture_count),
      required_sampler_count_(required_sampler_count)
{}

uint32_t Pipeline::thread_execution_width() const {
    return (uint32_t)state_->threadExecutionWidth();
}

uint32_t Pipeline::max_threads_per_threadgroup() const {
    return (uint32_t)state_->maxTotalThreadsPerThreadgroup();
}

MTL::Size Pipeline::compute_threadgroup_size(const std::array<uint32_t, 3>& grid) const {
    const uint32_t tew      = (uint32_t)state_->threadExecutionWidth();
    const uint32_t max_tot  = (uint32_t)state_->maxTotalThreadsPerThreadgroup();

    if (grid[1] == 1 && grid[2] == 1) {
        // Round down to a multiple of tew: the other two branches get this
        // for free (their width is tew itself), but max_tot isn't
        // guaranteed to be one. Pipelines are compiled with
        // threadGroupSizeIsMultipleOfThreadExecutionWidth=true, so every
        // branch here must actually uphold that invariant.
        uint32_t w = (max_tot / tew) * tew;
        return MTL::Size::Make(w > 0 ? w : tew, 1, 1);
    }
    if (grid[2] == 1) {
        uint32_t h = max_tot / tew;
        return MTL::Size::Make(tew, h > 0 ? h : 1, 1);
    }
    uint32_t w = tew;
    uint32_t h = std::max(1u, (uint32_t)std::sqrt((double)(max_tot / w)));
    uint32_t d = std::max(1u, max_tot / (w * h));
    return MTL::Size::Make(w, h, d);
}

std::pair<double, double> Pipeline::run(
    const std::vector<Buffer*>&    buffers,
    const std::vector<Texture*>&   textures,
    const std::vector<Sampler*>&   samplers,
    const std::array<uint32_t, 3>& grid,
    bool                           wait
) {
    if (buffers.size() < required_buffer_count_) {
        throw std::runtime_error(
            "Kernel reads buffer argument(s) up to index " +
            std::to_string(required_buffer_count_ - 1) + ", but only " +
            std::to_string(buffers.size()) +
            " buffer(s) were passed to run() -- an unbound buffer argument "
            "is undefined behavior in Metal, not a safe no-op."
        );
    }
    if (textures.size() < required_texture_count_) {
        throw std::runtime_error(
            "Kernel reads texture argument(s) up to index " +
            std::to_string(required_texture_count_ - 1) + ", but only " +
            std::to_string(textures.size()) +
            " texture(s) were passed to run() -- an unbound texture argument "
            "is undefined behavior in Metal, not a safe no-op."
        );
    }
    if (samplers.size() < required_sampler_count_) {
        throw std::runtime_error(
            "Kernel reads sampler argument(s) up to index " +
            std::to_string(required_sampler_count_ - 1) + ", but only " +
            std::to_string(samplers.size()) +
            " sampler(s) were passed to run() -- an unbound sampler argument "
            "is undefined behavior in Metal, not a safe no-op."
        );
    }

    PoolGuard guard;

    // Must retain references: with wait=False the caller can drop its last
    // Python reference to an input/output Buffer (e.g. reassigning `acc` in
    // an accumulate loop) before the GPU has actually finished the dispatch.
    // commandBuffer() makes Metal hold the MTL::Buffers alive until this
    // command buffer completes, regardless of what Python does in the
    // meantime.
    auto* cmd = queue_->commandBuffer();
    if (!cmd)
        throw std::runtime_error("Failed to create Metal command buffer");

    auto* encoder = cmd->computeCommandEncoder();
    if (!encoder)
        throw std::runtime_error("Failed to create compute encoder");

    encoder->setComputePipelineState(state_);

    for (size_t i = 0; i < buffers.size(); ++i)
        encoder->setBuffer(buffers[i]->mtl(), 0, i);
    for (size_t i = 0; i < textures.size(); ++i)
        encoder->setTexture(textures[i]->mtl(), i);
    for (size_t i = 0; i < samplers.size(); ++i)
        encoder->setSamplerState(samplers[i]->mtl(), i);

    MTL::Size grid_size         = MTL::Size::Make(grid[0], grid[1], grid[2]);
    MTL::Size threads_per_group = compute_threadgroup_size(grid);

    encoder->dispatchThreads(grid_size, threads_per_group);
    encoder->endEncoding();

    cmd->commit();

    if (wait) {
        cmd->waitUntilCompleted();
        if (cmd->status() == MTL::CommandBufferStatusError) {
            std::string err = cmd->error()
                ? cmd->error()->localizedDescription()->utf8String()
                : "Unknown GPU error";
            throw std::runtime_error("GPU execution failed: " + err);
        }
        return {cmd->GPUStartTime(), cmd->GPUEndTime()};
    }
    return {0.0, 0.0};
}

} // namespace mtlpy
