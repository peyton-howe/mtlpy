#pragma once
#include <Metal/Metal.hpp>

namespace mtlpy {

// Wraps MTL::CaptureScope: a labeled begin/end marker Xcode's GPU debugger
// (or a .gputrace file opened later in Xcode) shows as a named region in
// its capture timeline -- see Device::create_capture_scope()/start_capture().
// begin_scope()/end_scope() are harmless no-ops if nothing is actually
// capturing right now (neither Device::start_capture() nor Xcode's own
// capture button/scheme setting) -- Metal only records scope boundaries
// while a capture is in progress.
class CaptureScope {
public:
    // Takes ownership of an already-created MTL::CaptureScope* (Device::
    // create_capture_scope() makes it via MTLCaptureManager::newCaptureScope(),
    // a "new"-prefixed factory method -- already +1, same convention as
    // Queue/Event/Fence's constructors, no extra retain needed here).
    explicit CaptureScope(MTL::CaptureScope* scope);
    ~CaptureScope();

    void begin_scope();
    void end_scope();

    MTL::CaptureScope* mtl() const { return scope_; }

private:
    MTL::CaptureScope* scope_;  // +1 owned
};

} // namespace mtlpy
