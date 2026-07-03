#pragma once
#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mtlpy {

class PipelineCache {
public:
    PipelineCache() = default;
    ~PipelineCache();

    MTL::ComputePipelineState* get_or_create(
        MTL::Device*       device,
        const std::string& source,
        const std::string& function_name
    );

private:
    std::unordered_map<std::string, MTL::ComputePipelineState*> cache_;
    std::mutex mutex_;
};

} // namespace mtlpy
