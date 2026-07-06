#pragma once
#include <Metal/Metal.hpp>
#include <array>
#include <vector>

namespace mtlpy {

class Buffer;

class Pipeline {
public:
    Pipeline(MTL::ComputePipelineState* state, MTL::CommandQueue* queue,
             uint32_t required_buffer_count = 0);

    void run(
        const std::vector<Buffer*>&      buffers,
        const std::array<uint32_t, 3>&   grid,
        bool                             wait
    );

    uint32_t thread_execution_width()       const;
    uint32_t max_threads_per_threadgroup()  const;

private:
    MTL::ComputePipelineState* state_;  // non-owning; owned by PipelineCache
    MTL::CommandQueue*         queue_;  // non-owning; owned by Device
    uint32_t                   required_buffer_count_;

    MTL::Size compute_threadgroup_size(const std::array<uint32_t, 3>& grid) const;
};

} // namespace mtlpy
