#pragma once
#include <Metal/Metal.hpp>
#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace mtlpy {

class Buffer;
class Texture;
class Sampler;

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
    // threadgroup, if given, overrides compute_threadgroup_size()'s
    // heuristic with an exact threads-per-threadgroup size -- e.g. to match
    // a tile size a kernel's threadgroup-memory usage depends on. Must
    // satisfy the same two constraints Metal itself enforces: total threads
    // (w*h*d) <= max_threads_per_threadgroup, and a multiple of
    // thread_execution_width (this pipeline is compiled with
    // threadGroupSizeIsMultipleOfThreadExecutionWidth=true -- see
    // PipelineCache::get_or_create -- so violating that is undefined
    // behavior in Metal, not a safe-but-slow fallback). Both are validated
    // here with a clear error rather than left to fail inside the Metal
    // validation layer. std::nullopt (the default) keeps the existing
    // auto-computed behavior.
    //
    // Returns (gpu_start, gpu_end) in seconds, from MTLCommandBuffer's
    // GPUStartTime/GPUEndTime -- pure device-side execution time, excluding
    // CPU-side encoding/dispatch overhead and (when wait=true) the
    // waitUntilCompleted() latency itself. Only valid when wait=true (the
    // command buffer hasn't necessarily even started on the GPU, let alone
    // finished, until it completes); (0, 0) when wait=false.
    std::pair<double, double> run(
        const std::vector<Buffer*>&      buffers,
        const std::vector<Texture*>&     textures,
        const std::vector<Sampler*>&     samplers,
        const std::array<uint32_t, 3>&   grid,
        bool                             wait,
        const std::optional<std::array<uint32_t, 3>>& threadgroup = std::nullopt
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
    MTL::Size validate_threadgroup_size(const std::array<uint32_t, 3>& threadgroup) const;
};

} // namespace mtlpy
