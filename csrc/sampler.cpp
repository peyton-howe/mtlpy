#include "sampler.h"
#include <stdexcept>

namespace mtlpy {

Sampler::Sampler(MTL::Device* device, bool linear, bool repeat) {
    auto* desc = MTL::SamplerDescriptor::alloc()->init();

    auto filter = linear ? MTL::SamplerMinMagFilterLinear : MTL::SamplerMinMagFilterNearest;
    desc->setMinFilter(filter);
    desc->setMagFilter(filter);

    auto address = repeat ? MTL::SamplerAddressModeRepeat : MTL::SamplerAddressModeClampToEdge;
    desc->setSAddressMode(address);
    desc->setTAddressMode(address);
    desc->setRAddressMode(address);

    state_ = device->newSamplerState(desc);
    desc->release();

    if (!state_)
        throw std::runtime_error("Failed to create Metal sampler state");
}

Sampler::~Sampler() {
    state_->release();
}

} // namespace mtlpy
