#pragma once
#include <Metal/Metal.hpp>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

namespace mtlpy {

class Event;

// Lets multiple Pipeline::run() dispatches share one MTL::CommandBuffer
// (and one MTL::ComputeCommandEncoder) instead of each paying its own
// command-buffer-create + commit + waitUntilCompleted round trip -- see
// Device::create_command_buffer(). Encode dispatches into it via
// Pipeline::run(..., this), then call commit() once at the end.
//
// Thread safety: Pipeline::run and CommandBuffer::commit both release the
// GIL for their whole call (needed so a wait=true dispatch doesn't block
// every other Python thread), which means two Python threads sharing one
// CommandBuffer can genuinely execute inside this class concurrently --
// unlike a plain (non-batched) Pipeline::run() call, whose cmd/encoder are
// private stack locals with no such exposure. encode()/commit() hold
// mutex_ for their entire body (not just around encoder creation) because
// MTLComputeCommandEncoder itself isn't safe for concurrent use from
// multiple threads, not just its lazy creation.
class CommandBuffer {
public:
    explicit CommandBuffer(MTL::CommandQueue* queue);
    ~CommandBuffer();

    // Runs `fn` with this CommandBuffer's shared encoder (created lazily on
    // first use -- a CommandBuffer that ends up encoding nothing shouldn't
    // pay for one), holding mutex_ for the whole call so concurrent callers
    // serialize instead of racing. Sets the encoder's pipeline state first,
    // skipping the call if it's unchanged from the last dispatch encoded
    // into this same CommandBuffer (consecutive dispatches reusing one
    // Pipeline -- the common batching case -- would otherwise reissue an
    // identical, redundant setComputePipelineState on every dispatch).
    //
    // Throws (and marks this CommandBuffer failed -- see below) if called
    // after commit(), or if fn throws.
    void encode(MTL::ComputePipelineState* state,
                const std::function<void(MTL::ComputeCommandEncoder*)>& fn);

    // Encodes a command-buffer-level wait on `event` reaching `value` --
    // every command encoded into this CommandBuffer *after* this call (via
    // encode(), or a later wait/signal) will not start on the GPU until the
    // event does. Ends this CommandBuffer's currently-open encoder first if
    // there is one (MTLCommandBuffer::encodeWaitForEvent is only legal
    // between encoders, not while one is active) -- a later encode() call
    // transparently opens a fresh one, so batching more dispatches into this
    // CommandBuffer after a wait still works exactly as before, just with
    // the wait spliced into the command stream at this point. Throws under
    // the same conditions as encode() (already committed / already failed).
    void encode_wait_for_event(Event* event, uint64_t value);

    // The producer side of encode_wait_for_event(): signals `event` to
    // `value` once every command encoded into this CommandBuffer *before*
    // this call has completed on the GPU. Same encoder-splicing behavior as
    // encode_wait_for_event() above.
    void encode_signal_event(Event* event, uint64_t value);

    // Marks this CommandBuffer failed without going through encode() --
    // for Pipeline::run to call when it throws *before* ever reaching
    // encode() (e.g. its buffer/texture/sampler count validation), so a
    // batch is poisoned by any failed dispatch, not just one that made it
    // as far as actually touching the encoder.
    void mark_failed();

    // Ends encoding (if encode() was ever called) and commits. Returns
    // (gpu_start, gpu_end) in seconds when wait=true (MTLCommandBuffer's
    // GPUStartTime/GPUEndTime, covering every dispatch encoded into this
    // CommandBuffer combined -- there's no per-dispatch timing once
    // dispatches share a command buffer), (0, 0) otherwise. Throws if
    // called more than once, or if a prior encode() call on this
    // CommandBuffer threw (a batch that failed partway through encoding is
    // incomplete/inconsistent -- refusing to commit it here makes "a failed
    // dispatch discards the whole batch" a property of this class itself,
    // not just the Python context manager's __exit__, which only protects
    // the `with device.command_buffer() as cb:` usage and not the bare
    // `cb = device.command_buffer()` pattern the README documents as
    // equally supported).
    std::pair<double, double> commit(bool wait);

private:
    // Throws if this CommandBuffer can no longer accept new work (already
    // committed, or a previous encode()/encode_wait_for_event()/
    // encode_signal_event() call already failed) -- shared by all three, so
    // the "already committed"/"a previous dispatch failed" wording only
    // exists once. Deliberately separate from commit()'s own guard just
    // below it, which throws different (commit-specific) wording for the
    // same two conditions. Callers must hold mutex_.
    void check_encodable() const;

    // Ends and releases encoder_ if one is currently open, leaving this
    // CommandBuffer ready for either a fresh encode() or cmd_->commit().
    // Shared by encode_wait_for_event()/encode_signal_event() (both need to
    // splice a command-buffer-level call between encoders) -- callers must
    // hold mutex_.
    void close_encoder();

    std::mutex                  mutex_;
    MTL::CommandBuffer*         cmd_;      // +1 owned (explicitly retained)
    MTL::ComputeCommandEncoder* encoder_ = nullptr;  // +1 owned once created
    MTL::ComputePipelineState*  last_state_ = nullptr;  // non-owning; last state set on encoder_
    bool                        committed_ = false;
    bool                        failed_ = false;
};

} // namespace mtlpy
