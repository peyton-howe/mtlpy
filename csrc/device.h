#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace mtlpy {

class Buffer;
class Pipeline;
class PipelineCache;
class Texture;
class Sampler;

class Device {
public:
    // index < 0 (the default) uses CreateSystemDefaultDevice(); index >= 0
    // selects that position in available_device_names()/CopyAllDevices(),
    // for multi-GPU machines.
    explicit Device(int index = -1);
    ~Device();

    Buffer*   create_buffer(size_t size_bytes);
    Pipeline* compile(const std::string& source, const std::string& function_name);

    // dims is 1/2/3 (see Texture); pixel_format is a raw MTL::PixelFormat
    // value (see src/mtlpy/utils.py's pixel format table).
    Texture* create_texture(uint32_t dims, uint32_t pixel_format,
                             uint32_t width, uint32_t height, uint32_t depth);
    Sampler* create_sampler(bool linear, bool repeat);

    uint32_t max_threads_per_threadgroup() const;
    void     flush_cache();

    static std::vector<std::string> available_device_names();

private:
    MTL::Device*       device_;
    MTL::CommandQueue* queue_;
    PipelineCache*     cache_;
};

} // namespace mtlpy
