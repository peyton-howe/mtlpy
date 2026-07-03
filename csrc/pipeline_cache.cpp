#include "pipeline_cache.h"
#include <stdexcept>
#include <string>

namespace mtlpy {

PipelineCache::~PipelineCache() {
    for (auto& [key, state] : cache_)
        state->release();
}

MTL::ComputePipelineState* PipelineCache::get_or_create(
    MTL::Device*       device,
    const std::string& source,
    const std::string& function_name
) {
    std::string key = source + '\0' + function_name;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;

    NS::Error* error = nullptr;
    auto* src = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    auto* library = device->newLibrary(src, nullptr, &error);
    if (!library) {
        throw std::runtime_error(
            std::string("Shader compilation failed: ") +
            error->localizedDescription()->utf8String()
        );
    }

    auto* fname    = NS::String::string(function_name.c_str(), NS::UTF8StringEncoding);
    auto* function = library->newFunction(fname);
    library->release();

    if (!function)
        throw std::runtime_error("Function not found in shader: " + function_name);

    auto* state = device->newComputePipelineState(function, &error);
    function->release();

    if (!state) {
        throw std::runtime_error(
            std::string("Failed to create pipeline state: ") +
            error->localizedDescription()->utf8String()
        );
    }

    cache_[key] = state;
    return state;
}

} // namespace mtlpy
