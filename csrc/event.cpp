#include "event.h"
#include "pool_guard.h"
#include <stdexcept>

namespace mtlpy {

Event::Event(MTL::Device* device) {
    PoolGuard guard;
    event_ = device->newEvent();
    if (!event_)
        throw std::runtime_error("Failed to create Metal event");
}

Event::~Event() {
    PoolGuard guard;
    event_->release();
}

SharedEventHandle::SharedEventHandle(MTL::SharedEventHandle* handle)
    : handle_(handle)
{
    if (!handle_)
        throw std::runtime_error("Failed to create Metal shared event handle");
}

SharedEventHandle::~SharedEventHandle() {
    PoolGuard guard;
    handle_->release();
}

SharedEvent::SharedEvent(MTL::Device* device)
    : Event(static_cast<MTL::Event*>(nullptr))
{
    PoolGuard guard;
    event_ = device->newSharedEvent();
    if (!event_)
        throw std::runtime_error("Failed to create Metal shared event");
}

SharedEvent::SharedEvent(MTL::Device* device, SharedEventHandle* handle)
    : Event(static_cast<MTL::Event*>(nullptr))
{
    PoolGuard guard;
    event_ = device->newSharedEvent(handle->mtl());
    if (!event_)
        throw std::runtime_error(
            "Failed to import Metal shared event from handle");
}

void SharedEvent::signal(uint64_t value) {
    mtl_shared()->setSignaledValue(value);
}

uint64_t SharedEvent::signaled_value() const {
    return mtl_shared()->signaledValue();
}

bool SharedEvent::wait(uint64_t value, uint64_t timeout_ms) {
    return mtl_shared()->waitUntilSignaledValue(value, timeout_ms);
}

SharedEventHandle* SharedEvent::new_shared_event_handle() const {
    PoolGuard guard;
    auto* handle = mtl_shared()->newSharedEventHandle();
    if (!handle)
        throw std::runtime_error("Failed to export Metal shared event handle");
    return new SharedEventHandle(handle);
}

} // namespace mtlpy
