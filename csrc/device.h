#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>
#include <string>

namespace mtlpy {

class Buffer;
class Pipeline;
class PipelineCache;

class Device {
public:
    Device();
    ~Device();

    Buffer*   create_buffer(size_t size_bytes);
    Pipeline* compile(const std::string& source, const std::string& function_name);

    uint32_t max_threads_per_threadgroup() const;

private:
    MTL::Device*       device_;
    MTL::CommandQueue* queue_;
    PipelineCache*     cache_;
};

} // namespace mtlpy
