#include "pipeline.h"
#include "buffer.h"
#include <cmath>
#include <stdexcept>

namespace mtlpy {

Pipeline::Pipeline(MTL::ComputePipelineState* state, MTL::CommandQueue* queue)
    : state_(state), queue_(queue)
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
        return MTL::Size::Make(max_tot, 1, 1);
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

void Pipeline::run(
    const std::vector<Buffer*>&    buffers,
    const std::array<uint32_t, 3>& grid,
    bool                           wait
) {
    auto* cmd = queue_->commandBuffer();
    if (!cmd)
        throw std::runtime_error("Failed to create Metal command buffer");

    auto* encoder = cmd->computeCommandEncoder();
    if (!encoder) {
        cmd->release();
        throw std::runtime_error("Failed to create compute encoder");
    }

    encoder->setComputePipelineState(state_);

    for (size_t i = 0; i < buffers.size(); ++i)
        encoder->setBuffer(buffers[i]->mtl(), 0, i);

    MTL::Size grid_size         = MTL::Size::Make(grid[0], grid[1], grid[2]);
    MTL::Size threads_per_group = compute_threadgroup_size(grid);

    encoder->dispatchThreads(grid_size, threads_per_group);
    encoder->endEncoding();
    encoder->release();

    cmd->commit();

    if (wait) {
        cmd->waitUntilCompleted();
        if (cmd->status() == MTL::CommandBufferStatusError) {
            std::string err = cmd->error()
                ? cmd->error()->localizedDescription()->utf8String()
                : "Unknown GPU error";
            cmd->release();
            throw std::runtime_error("GPU execution failed: " + err);
        }
    }

    cmd->release();
}

} // namespace mtlpy
