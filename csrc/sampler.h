#pragma once
#include <Metal/Metal.hpp>

namespace mtlpy {

// A deliberately small wrapper: linear-vs-nearest filtering and
// repeat-vs-clamp addressing (applied uniformly across s/t/r) cover the
// common cases without exposing every MTL::SamplerDescriptor knob, matching
// this library's existing thin-wrapper philosophy elsewhere (see e.g.
// Buffer, which doesn't expose every MTL::ResourceOptions bit either).
class Sampler {
public:
    Sampler(MTL::Device* device, bool linear, bool repeat);
    ~Sampler();

    MTL::SamplerState* mtl() const { return state_; }

private:
    MTL::SamplerState* state_;
};

} // namespace mtlpy
