#pragma once
#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mtlpy {

class PipelineCache {
public:
    // device is non-owning; used only to open/create the on-disk binary
    // archive that lets compiled pipelines survive across process launches.
    explicit PipelineCache(MTL::Device* device);
    ~PipelineCache();

    MTL::ComputePipelineState* get_or_create(
        MTL::Device*       device,
        const std::string& source,
        const std::string& function_name
    );

private:
    std::unordered_map<std::string, MTL::ComputePipelineState*> cache_;
    std::mutex mutex_;

    // May be null if the archive couldn't be opened/created; the cache
    // still works in that case, just without cross-process persistence.
    MTL::BinaryArchive* archive_ = nullptr;
    std::string         archive_path_;
};

} // namespace mtlpy
