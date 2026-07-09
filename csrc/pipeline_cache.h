#pragma once
#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mtlpy {

struct CachedPipeline {
    MTL::ComputePipelineState* state;
    // One past the highest active argument index the shader reads, per
    // binding namespace (from Metal's reflection info) -- i.e. the minimum
    // number of buffers/textures/samplers Pipeline::run() must be given.
    // Address-space qualifier doesn't matter for buffers: both `device` and
    // `constant` parameters report as ArgumentTypeBuffer.
    uint32_t required_buffer_count;
    uint32_t required_texture_count;
    uint32_t required_sampler_count;
};

class PipelineCache {
public:
    // device is non-owning; used only to open/create the on-disk binary
    // archive that lets compiled pipelines survive across process launches.
    explicit PipelineCache(MTL::Device* device);
    ~PipelineCache();

    CachedPipeline get_or_create(
        MTL::Device*       device,
        const std::string& source,
        const std::string& function_name
    );

    // Serialize the on-disk binary archive now, without waiting for the
    // destructor -- lets a long-running process checkpoint newly-compiled
    // pipelines periodically instead of only at exit (whose GC/finalizer
    // timing isn't deterministic).
    void flush();

private:
    std::unordered_map<std::string, CachedPipeline> cache_;
    std::mutex mutex_;

    // May be null if the archive couldn't be opened/created; the cache
    // still works in that case, just without cross-process persistence.
    MTL::BinaryArchive* archive_ = nullptr;
    std::string         archive_path_;
};

} // namespace mtlpy
