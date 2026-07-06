#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace mtlpy {

class Buffer;
class Pipeline;
class PipelineCache;

class Device {
public:
    // index < 0 (the default) uses CreateSystemDefaultDevice(); index >= 0
    // selects that position in available_device_names()/CopyAllDevices(),
    // for multi-GPU machines.
    explicit Device(int index = -1);
    ~Device();

    Buffer*   create_buffer(size_t size_bytes);
    Pipeline* compile(const std::string& source, const std::string& function_name);

    uint32_t max_threads_per_threadgroup() const;
    void     flush_cache();

    static std::vector<std::string> available_device_names();

private:
    MTL::Device*       device_;
    MTL::CommandQueue* queue_;
    PipelineCache*     cache_;
};

} // namespace mtlpy
