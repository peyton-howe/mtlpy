#include "pipeline.h"
#include "buffer.h"
#include "command_buffer.h"
#include "metal_error.h"
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
    bool                           wait,
    CommandBuffer*                 external_cb
) {
    // PoolGuard covers both branches below (not just the self-contained
    // one) -- it's stack-scoped to this single function call either way,
    // so nesting is always safe regardless of which branch runs (unlike
    // CommandBuffer's own pool problem, which came from a pool spanning
    // *multiple* separate calls, not from being used within one).
    PoolGuard guard;

    try {
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

        MTL::Size grid_size         = MTL::Size::Make(grid[0], grid[1], grid[2]);
        MTL::Size threads_per_group = compute_threadgroup_size(grid);

        // Binds buffers/textures/samplers and dispatches -- does NOT set
        // the pipeline state, unlike the old single combined lambda: the
        // batched path (CommandBuffer::encode) decides whether that call is
        // even needed (skipped when consecutive dispatches into the same
        // CommandBuffer reuse the same Pipeline), so it's pulled out to be
        // handled explicitly by each branch below instead.
        auto bind_resources_and_dispatch = [&](MTL::ComputeCommandEncoder* encoder) {
            for (size_t i = 0; i < buffers.size(); ++i)
                encoder->setBuffer(buffers[i]->mtl(), 0, i);
            for (size_t i = 0; i < textures.size(); ++i)
                encoder->setTexture(textures[i]->mtl(), i);
            for (size_t i = 0; i < samplers.size(); ++i)
                encoder->setSamplerState(samplers[i]->mtl(), i);
            encoder->dispatchThreads(grid_size, threads_per_group);
        };

        if (external_cb) {
            // Shared command buffer: bind + dispatch into its encoder, but
            // don't end encoding or commit -- the caller does that once via
            // CommandBuffer::commit() after encoding every dispatch it
            // wants batched together. Metal's own encoder-side setBuffer/
            // setTexture retain the resources for us, same guarantee the
            // self-contained path below relies on for wait=false.
            external_cb->encode(state_, bind_resources_and_dispatch);
            return {0.0, 0.0};
        }

        // Must retain references: with wait=False the caller can drop its
        // last Python reference to an input/output Buffer (e.g. reassigning
        // `acc` in an accumulate loop) before the GPU has actually finished
        // the dispatch. commandBuffer() makes Metal hold the MTL::Buffers
        // alive until this command buffer completes, regardless of what
        // Python does in the meantime.
        auto* cmd = queue_->commandBuffer();
        if (!cmd)
            throw std::runtime_error("Failed to create Metal command buffer");

        auto* encoder = cmd->computeCommandEncoder();
        if (!encoder)
            throw std::runtime_error("Failed to create compute encoder");

        encoder->setComputePipelineState(state_);
        bind_resources_and_dispatch(encoder);
        encoder->endEncoding();

        cmd->commit();

        if (wait) {
            cmd->waitUntilCompleted();
            throw_if_command_buffer_error(cmd, "GPU execution");
            return {cmd->GPUStartTime(), cmd->GPUEndTime()};
        }
        return {0.0, 0.0};
    } catch (...) {
        // Only meaningful for the batched path (encode() already marks
        // external_cb failed itself if the exception came from inside it --
        // this additionally covers exceptions thrown *before* encode() was
        // ever reached, e.g. the buffer/texture/sampler count validation
        // above, so any failed dispatch poisons the whole batch, not just
        // ones that made it as far as touching the encoder).
        if (external_cb)
            external_cb->mark_failed();
        throw;
    }
}

} // namespace mtlpy
