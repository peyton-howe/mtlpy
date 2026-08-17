#include "pipeline_cache.h"
#include "pool_guard.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace mtlpy {

namespace {

namespace fs = std::filesystem;

// Where compiled pipelines are persisted so a fresh Python process doesn't
// pay full shader-compile latency again. Best-effort: if $HOME isn't set or
// the directory can't be created, callers just don't get cross-process
// caching (see PipelineCache::PipelineCache).
std::string default_archive_path() {
    const char* home = std::getenv("HOME");
    fs::path dir = home ? fs::path(home) / "Library" / "Caches" / "mtlpy"
                         : fs::temp_directory_path() / "mtlpy";

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
        return "";

    return (dir / "pipelines.metallib").string();
}

NS::URL* url_for(const std::string& path) {
    return NS::URL::fileURLWithPath(
        NS::String::string(path.c_str(), NS::UTF8StringEncoding));
}

} // namespace

PipelineCache::PipelineCache(MTL::Device* device, const std::optional<std::string>& cache_path)
    : archive_path_(cache_path.has_value() ? *cache_path : default_archive_path())
{
    // Empty either because cache_path == "" was passed explicitly (on-disk
    // caching disabled) or default_archive_path() couldn't determine a
    // location (e.g. $HOME unset) -- either way, archive_ just stays null
    // and get_or_create() compiles from source every time.
    if (archive_path_.empty())
        return;

    PoolGuard guard;

    auto* descriptor = MTL::BinaryArchiveDescriptor::alloc()->init();
    if (fs::exists(archive_path_))
        descriptor->setUrl(url_for(archive_path_));

    NS::Error* error = nullptr;
    archive_ = device->newBinaryArchive(descriptor, &error);
    descriptor->release();
    // A missing or corrupt archive file is not fatal to Device construction
    // -- archive_ just stays null and get_or_create() compiles from source
    // every time, same as before this change.
}

PipelineCache::~PipelineCache() {
    PoolGuard guard;

    for (auto& [key, cached] : cache_) {
        cached.state->release();
        cached.function->release();
    }

    if (archive_) {
        NS::Error* error = nullptr;
        archive_->serializeToURL(url_for(archive_path_), &error);
        archive_->release();
    }
}

void PipelineCache::flush(const std::optional<std::string>& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!archive_)
        return;
    PoolGuard guard;
    NS::Error* error = nullptr;
    archive_->serializeToURL(url_for(path.has_value() ? *path : archive_path_), &error);
    // Best-effort, same as the destructor's serialize call -- a failed
    // flush (e.g. disk full) just means the next process recompiles from
    // source, not a reason to raise from what's meant to be a cheap,
    // periodic checkpoint.
}

size_t PipelineCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

void PipelineCache::register_in_archive(MTL::BinaryArchive* archive, MTL::Function* function) {
    if (!archive)
        return;
    PoolGuard guard;
    auto* descriptor = MTL::ComputePipelineDescriptor::alloc()->init();
    descriptor->setComputeFunction(function);
    descriptor->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
    NS::Error* error = nullptr;
    archive->addComputePipelineFunctions(descriptor, &error);
    descriptor->release();
}

CachedPipeline PipelineCache::get_or_create(
    MTL::Device*        device,
    const std::string&  source,
    const std::string&  function_name,
    MTL::BinaryArchive* extra_archive
) {
    std::string key = source + '\0' + function_name;

    // unique_lock (not lock_guard): the cache-hit branch below explicitly
    // unlocks before register_in_archive()'s Metal call -- mutex_ only ever
    // needs to protect cache_ itself, not an addComputePipelineFunctions
    // call against a caller-supplied archive, so there's no reason for that
    // call to serialize against unrelated compiles/lookups on this cache.
    std::unique_lock<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        CachedPipeline cached = it->second;
        lock.unlock();
        register_in_archive(extra_archive, cached.function);
        return cached;
    }

    PoolGuard guard;

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

    auto* descriptor = MTL::ComputePipelineDescriptor::alloc()->init();
    descriptor->setComputeFunction(function);
    // function is NOT released here (unlike library, which nothing needs
    // past this point): descriptor->setComputeFunction() takes its own
    // internal +1 for the descriptor's own use, same as always, but this
    // +1 (from newFunction() above) is kept as CachedPipeline::function --
    // see the struct's own doc comment for why. Released either in the
    // failure path just below, or -- once state creation actually succeeds
    // -- in PipelineCache's destructor, alongside cached.state.

    // compute_threadgroup_size() in Pipeline always dispatches this pipeline
    // with threadgroup sizes that are multiples of threadExecutionWidth,
    // so we can tell the compiler that up front and let it skip generating
    // boundary-check code it would otherwise need for arbitrary threadgroup
    // sizes.
    descriptor->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);

    if (archive_) {
        auto* archives = NS::Array::array(archive_);
        descriptor->setBinaryArchives(archives);
    }

    MTL::ComputePipelineReflection* reflection = nullptr;
    auto* state = device->newComputePipelineState(
        descriptor, MTL::PipelineOptionArgumentInfo, &reflection, &error);

    if (!state && archive_) {
        // The archive may hold a binary for a different GPU family/driver
        // than the one we're running on now; retry once without it rather
        // than failing outright.
        descriptor->setBinaryArchives(nullptr);
        error = nullptr;
        state = device->newComputePipelineState(
            descriptor, MTL::PipelineOptionArgumentInfo, &reflection, &error);
    }

    if (!state) {
        descriptor->release();
        function->release();  // see the comment above -- never got to CachedPipeline
        throw std::runtime_error(
            std::string("Failed to create pipeline state: ") +
            error->localizedDescription()->utf8String()
        );
    }

    if (archive_) {
        NS::Error* archive_error = nullptr;
        archive_->addComputePipelineFunctions(descriptor, &archive_error);
    }
    register_in_archive(extra_archive, function);

    descriptor->release();

    // One past the highest active argument index the shader reads, per
    // binding namespace -- both `device` and `constant` buffer parameters
    // report as ArgumentTypeBuffer, so that covers both without needing to
    // distinguish them.
    uint32_t required_buffer_count  = 0;
    uint32_t required_texture_count = 0;
    uint32_t required_sampler_count = 0;
    if (reflection) {
        auto* args = reflection->arguments();
        for (NS::UInteger i = 0; i < args->count(); ++i) {
            auto* arg = args->object<MTL::Argument>(i);
            if (!arg->isActive())
                continue;
            uint32_t index = (uint32_t)arg->index() + 1;
            switch (arg->type()) {
                case MTL::ArgumentTypeBuffer:
                    required_buffer_count = std::max(required_buffer_count, index);
                    break;
                case MTL::ArgumentTypeTexture:
                    required_texture_count = std::max(required_texture_count, index);
                    break;
                case MTL::ArgumentTypeSampler:
                    required_sampler_count = std::max(required_sampler_count, index);
                    break;
                default:
                    break;
            }
        }
    }

    CachedPipeline cached{state, function, required_buffer_count, required_texture_count, required_sampler_count};
    cache_[key] = cached;
    return cached;
}

} // namespace mtlpy
