#pragma once
#include <Metal/Metal.hpp>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace mtlpy {

struct CachedPipeline {
    MTL::ComputePipelineState* state;
    // Kept alive (+1 owned, alongside state, for as long as this cache
    // entry lives -- see PipelineCache's destructor) so a later
    // get_or_create() call with a different extra_archive can register
    // this already-compiled pipeline into it without ever touching
    // `source` again -- see register_in_archive().
    MTL::Function*              function;
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
    //
    // cache_path overrides where that on-disk archive lives: std::nullopt
    // (the default) computes the usual ~/Library/Caches/mtlpy/pipelines.metallib
    // location (see default_archive_path() in pipeline_cache.cpp); an empty
    // string explicitly disables on-disk caching for this Device entirely
    // (pipelines are still deduped in-memory for this process, just never
    // written to/read from disk); any other string is used verbatim as the
    // archive file's path.
    explicit PipelineCache(MTL::Device* device,
                            const std::optional<std::string>& cache_path = std::nullopt);
    ~PipelineCache();

    // extra_archive (default nullptr) additionally registers the resulting
    // pipeline into that BinaryArchive -- see Device::compile's archive
    // param. Works whether this call recompiles from source or hits the
    // in-memory cache: on a hit, the cached MTL::Function is reused to
    // build a fresh descriptor for extra_archive without recompiling
    // `source`.
    CachedPipeline get_or_create(
        MTL::Device*         device,
        const std::string&   source,
        const std::string&   function_name,
        MTL::BinaryArchive*  extra_archive = nullptr
    );

    // Serialize the on-disk binary archive now, without waiting for the
    // destructor -- lets a long-running process checkpoint newly-compiled
    // pipelines periodically instead of only at exit (whose GC/finalizer
    // timing isn't deterministic). path (default std::nullopt) overrides
    // the destination for this call only, without changing where future
    // flush() calls (or the destructor) write to.
    void flush(const std::optional<std::string>& path = std::nullopt);

    size_t size() const;
    const std::string& path() const { return archive_path_; }

private:
    // Registers `function` (already compiled, either just now or from a
    // prior get_or_create() call) into `archive` as its own
    // MTL::ComputePipelineDescriptor -- shared by both branches of
    // get_or_create() below. No-op if archive is null. Best-effort: a
    // failed add isn't surfaced as an exception (the caller already has a
    // working compiled Pipeline regardless of whether this succeeds); only
    // an explicit save()/flush() failing is something a caller needs to
    // react to.
    void register_in_archive(MTL::BinaryArchive* archive, MTL::Function* function);

    std::unordered_map<std::string, CachedPipeline> cache_;
    mutable std::mutex mutex_;

    // May be null if the archive couldn't be opened/created, or on-disk
    // caching was explicitly disabled (cache_path == ""); the cache still
    // works in that case, just without cross-process persistence.
    MTL::BinaryArchive* archive_ = nullptr;
    std::string         archive_path_;
};

} // namespace mtlpy
