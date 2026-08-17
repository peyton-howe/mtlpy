#pragma once
#include <Metal/Metal.hpp>

namespace mtlpy {

// A second (or third, ...) MTL::CommandQueue beyond the one every Device
// already owns for its self-contained dispatch/blit paths (see device.h) --
// lets independent streams of work run concurrently on the GPU, synchronized
// only where an Event/Fence explicitly says so, instead of everything
// funneling through one serial queue. Only usable via the batched dispatch
// path (Device::create_command_buffer(queue), Pipeline::run(..., cb=...)):
// a Pipeline's *self-contained* dispatch (no CommandBuffer given) always
// targets the Device's own default queue, the one it was compiled against.
class Queue {
public:
    // Takes ownership of an already-created MTL::CommandQueue* (Device::
    // create_queue() makes it via device_->newCommandQueue()).
    explicit Queue(MTL::CommandQueue* queue);
    ~Queue();

    MTL::CommandQueue* mtl() const { return queue_; }

private:
    MTL::CommandQueue* queue_;  // +1 owned
};

} // namespace mtlpy
