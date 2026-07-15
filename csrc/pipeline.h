#pragma once
#include <Metal/Metal.hpp>
#include <array>
#include <utility>
#include <vector>

namespace mtlpy {

class Buffer;
class Texture;
class Sampler;
class CommandBuffer;

class Pipeline {
public:
    Pipeline(MTL::ComputePipelineState* state, MTL::CommandQueue* queue,
             uint32_t required_buffer_count = 0,
             uint32_t required_texture_count = 0,
             uint32_t required_sampler_count = 0);

    // Buffers, textures, and samplers occupy independent binding namespaces
    // in Metal Shading Language ([[buffer(n)]] / [[texture(n)]] /
    // [[sampler(n)]]) -- each list here is bound by its own position, same
    // convention as buffers already used (list index i -> binding i).
    //
    // external_cb (default nullptr) -- when given, encodes this dispatch
    // into that CommandBuffer's shared encoder instead of creating and
    // committing its own command buffer, so multiple run() calls can batch
    // into one submission (see CommandBuffer). wait is ignored in that case
    // (the caller controls waiting via CommandBuffer::commit(wait) once,
    // after encoding every dispatch it wants batched together), and this
    // always returns (0, 0) -- per-dispatch GPU timing isn't meaningful
    // once dispatches share a command buffer, only CommandBuffer::commit()'s
    // combined timing is. If this call throws (e.g. the buffer/texture/
    // sampler count validation below fails), external_cb is marked failed
    // and any later commit() on it throws too -- a batch a dispatch failed
    // partway into is never silently partially submitted.
    //
    // Otherwise (external_cb == nullptr, the default), returns (gpu_start,
    // gpu_end) in seconds, from MTLCommandBuffer's GPUStartTime/GPUEndTime
    // -- pure device-side execution time, excluding CPU-side encoding/
    // dispatch overhead and (when wait=true) the waitUntilCompleted()
    // latency itself. Only valid when wait=true (the command buffer hasn't
    // necessarily even started on the GPU, let alone finished, until it
    // completes); (0, 0) when wait=false.
    std::pair<double, double> run(
        const std::vector<Buffer*>&      buffers,
        const std::vector<Texture*>&     textures,
        const std::vector<Sampler*>&     samplers,
        const std::array<uint32_t, 3>&   grid,
        bool                             wait,
        CommandBuffer*                   external_cb = nullptr
    );

    uint32_t thread_execution_width()       const;
    uint32_t max_threads_per_threadgroup()  const;

private:
    MTL::ComputePipelineState* state_;  // non-owning; owned by PipelineCache
    MTL::CommandQueue*         queue_;  // non-owning; owned by Device
    uint32_t                   required_buffer_count_;
    uint32_t                   required_texture_count_;
    uint32_t                   required_sampler_count_;

    MTL::Size compute_threadgroup_size(const std::array<uint32_t, 3>& grid) const;
};

} // namespace mtlpy
