#pragma once
#include <Metal/Metal.hpp>
#include <cstdint>

namespace mtlpy {

// Wraps MTL::Event: a GPU-side-only synchronization primitive for ordering
// work across separate MTL::CommandBuffers (see CommandBuffer::
// encode_signal_event/encode_wait_for_event) -- including buffers submitted
// to *different* MTL::CommandQueues, which is the one ordering guarantee a
// shared queue's own commit-order semantics don't give you for free (see
// Device::create_queue()). Cheaper than SharedEvent when nothing needs to
// read/wait on it CPU-side: there's no signaled-value readback and no
// cross-process handle, just GPU-to-GPU signal/wait.
class Event {
public:
    explicit Event(MTL::Device* device);
    virtual ~Event();

    MTL::Event* mtl() const { return event_; }

protected:
    // Lets SharedEvent's constructor install its own, already-created
    // MTL::SharedEvent* (itself an MTL::Event via Objective-C inheritance,
    // see event.cpp) instead of this base class creating a redundant plain
    // MTL::Event first.
    explicit Event(MTL::Event* already_owned) : event_(already_owned) {}

    MTL::Event* event_;  // +1 owned
};

// Wraps MTL::SharedEventHandle: an opaque, exportable reference to a
// SharedEvent (SharedEvent::new_shared_event_handle()) that another process
// can import via Device::create_shared_event_from_handle() to synchronize
// with this same event across process boundaries. Actually transporting the
// handle between processes (e.g. over an XPC connection -- MTLSharedEventHandle
// conforms to NSSecureCoding, so NSXPCConnection knows how to encode it) is
// the caller's responsibility; mtlpy only provides the create/export/import
// primitives, not an IPC channel of its own.
class SharedEventHandle {
public:
    explicit SharedEventHandle(MTL::SharedEventHandle* handle);
    ~SharedEventHandle();

    MTL::SharedEventHandle* mtl() const { return handle_; }

private:
    MTL::SharedEventHandle* handle_;  // +1 owned
};

// Wraps MTL::SharedEvent: like Event, but adds a CPU-visible uint64
// "signaled value" that can be set/read directly from the CPU
// (signal()/signaled_value) and blocked on from the CPU (wait()) -- the
// mechanism for CPU<->GPU handoff (e.g. the CPU calls signal() to unblock a
// command buffer's encoded wait_for_event(), or wait()s for a value the GPU
// side signals via CommandBuffer::encode_signal_event()). Also exportable
// via new_shared_event_handle() for cross-process use -- see
// SharedEventHandle.
class SharedEvent : public Event {
public:
    explicit SharedEvent(MTL::Device* device);
    SharedEvent(MTL::Device* device, SharedEventHandle* handle);

    void     signal(uint64_t value);
    uint64_t signaled_value() const;

    // Blocks the calling thread until signaled_value() reaches at least
    // `value`, or timeout_ms elapses. Returns false on timeout, true once
    // satisfied. Releases the GIL for the whole call (see bindings.cpp) --
    // same rationale as CommandBuffer::commit(wait=True).
    bool wait(uint64_t value, uint64_t timeout_ms);

    SharedEventHandle* new_shared_event_handle() const;

    MTL::SharedEvent* mtl_shared() const {
        // Legal downcast: MTL::SharedEvent publicly inherits MTL::Event
        // (NS::Referencing<SharedEvent, Event> : public Event), so event_
        // set by either constructor below always actually points at an
        // MTL::SharedEvent instance.
        return static_cast<MTL::SharedEvent*>(event_);
    }
};

} // namespace mtlpy
