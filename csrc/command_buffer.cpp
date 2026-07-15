#include "command_buffer.h"
#include "metal_error.h"
#include <stdexcept>

namespace mtlpy {

CommandBuffer::CommandBuffer(MTL::CommandQueue* queue) {
    cmd_ = queue->commandBuffer();
    if (!cmd_)
        throw std::runtime_error("Failed to create Metal command buffer");
    // commandBuffer() (unlike a "new"/"alloc"/"copy"-prefixed factory)
    // returns an autoreleased (+0) reference -- explicitly retaining it
    // converts that to a +1 reference we own directly, independent of
    // whatever autorelease pool happens to be active when this was called.
    // That matters here specifically because a CommandBuffer's lifetime
    // spans multiple separate Python calls (construction, one or more
    // Pipeline::run(cb=...) calls, then commit()) rather than a single C++
    // function body -- unlike Pipeline::run's stack-scoped PoolGuard (always
    // safely nested/LIFO by C++ scoping), a pool covering this object's
    // whole lifetime would have to be drained in the same order it and
    // every other live CommandBuffer's pool were created, which Python's
    // garbage collector gives no guarantee of (confirmed by testing:
    // multiple CommandBuffers destroyed out of creation order crashed with
    // "Command encoder released without endEncoding" / pool-nesting
    // violations). Manual retain/release sidesteps the ordering requirement
    // entirely, matching how Buffer/Texture/etc. already own their Metal
    // objects long-term.
    cmd_->retain();
}

CommandBuffer::~CommandBuffer() {
    // If commit() was never called (e.g. the Python context manager's
    // __exit__ deliberately skips it after an exception, to discard a
    // partially-encoded batch), an encoder() that was already created is
    // still an open MTLCommandEncoder -- Metal asserts/crashes if one is
    // released without endEncoding(), regardless of whether the owning
    // command buffer itself ever gets committed. Ending it here does not
    // submit any work (cmd_->commit() is never called on this path), it
    // only satisfies that requirement before the encoder is released.
    if (encoder_) {
        if (!committed_)
            encoder_->endEncoding();
        encoder_->release();
    }
    cmd_->release();
}

void CommandBuffer::encode(MTL::ComputePipelineState* state,
                            const std::function<void(MTL::ComputeCommandEncoder*)>& fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed_)
        throw std::runtime_error(
            "CommandBuffer already committed -- cannot encode more work into it");
    if (failed_)
        throw std::runtime_error(
            "A previous dispatch encoded into this CommandBuffer failed -- the batch "
            "is incomplete and cannot be committed; create a new CommandBuffer instead");
    try {
        if (!encoder_) {
            encoder_ = cmd_->computeCommandEncoder();
            if (!encoder_)
                throw std::runtime_error("Failed to create compute encoder");
            encoder_->retain();  // same +0 -> +1 reasoning as cmd_ above
        }
        if (state != last_state_) {
            encoder_->setComputePipelineState(state);
            last_state_ = state;
        }
        fn(encoder_);
    } catch (...) {
        failed_ = true;
        throw;
    }
}

void CommandBuffer::mark_failed() {
    std::lock_guard<std::mutex> lock(mutex_);
    failed_ = true;
}

std::pair<double, double> CommandBuffer::commit(bool wait) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed_)
        throw std::runtime_error("CommandBuffer already committed");
    if (failed_)
        throw std::runtime_error(
            "Cannot commit: a dispatch encoded into this CommandBuffer failed, so the "
            "batch is incomplete -- create a new CommandBuffer instead");
    committed_ = true;

    if (encoder_)
        encoder_->endEncoding();
    cmd_->commit();

    if (wait) {
        cmd_->waitUntilCompleted();
        throw_if_command_buffer_error(cmd_, "GPU execution");
        return {cmd_->GPUStartTime(), cmd_->GPUEndTime()};
    }
    return {0.0, 0.0};
}

} // namespace mtlpy
