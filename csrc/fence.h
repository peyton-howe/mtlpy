#pragma once
#include <Metal/Metal.hpp>

namespace mtlpy {

// Wraps MTL::Fence: a lightweight producer-consumer ordering primitive
// between two encoders in the *same* MTL::CommandQueue (either two encoders
// in one command buffer, or encoders split across separate command buffers
// on that queue) -- see Pipeline::run's wait_fences/signal_fences. Metal
// already auto-tracks resource hazards for the Buffers/Textures this
// library hands out (none are created with hazardTrackingModeUntracked), so
// a Fence is rarely required for plain correctness here; it exists as an
// explicit, lower-level tool for orderings that aren't implied by resource
// usage alone (e.g. a CPU-invisible side effect, or a future
// hazard-tracking-disabled resource). For ordering across *different*
// command queues, use Event instead (see event.h) -- Fence's guarantee is
// scoped to a single queue, unlike Event.
class Fence {
public:
    explicit Fence(MTL::Device* device);
    ~Fence();

    MTL::Fence* mtl() const { return fence_; }

private:
    MTL::Fence* fence_;  // +1 owned
};

} // namespace mtlpy
