#pragma once
#include <Metal/Metal.hpp>
#include <string>

namespace mtlpy {

// Wraps MTL::BinaryArchive: an explicit, user-managed precompiled-pipeline
// archive, independent of the implicit one every Device already maintains
// internally (see PipelineCache) -- for cases that one doesn't cover: a
// custom on-disk location, sharing one archive across several Devices in
// the same process, or building an archive meant to ship as a read-only
// asset with an app (compile everything once during a build step, save()
// the result, then load it back via Device::create_binary_archive(path) at
// runtime for fast first-launch pipeline creation with no shader
// recompilation).
//
// Passed to Device::compile(..., extra_archive) to have that specific
// compile additionally register into this archive, on top of whatever the
// Device's own internal cache already does -- see PipelineCache::
// get_or_create's extra_archive param.
class BinaryArchive {
public:
    // path.empty() creates a fresh, empty, in-memory-only archive (nothing
    // to open yet). A non-empty path that already exists on disk opens and
    // parses it (Metal fails outright if a URL is set but nothing exists
    // there, so this only sets the descriptor's URL when the file is
    // actually present -- same guard PipelineCache's constructor uses). A
    // non-empty path that doesn't exist yet also creates a fresh empty
    // archive, same as an empty path -- the file only comes into existence
    // once save() is actually called. Either way, `path` is remembered as
    // save()'s default destination.
    BinaryArchive(MTL::Device* device, const std::string& path);
    ~BinaryArchive();

    // Serializes this archive to disk at `path`, or (path.empty(), the
    // default) the path this archive was constructed with -- throws if
    // that was also empty (a fresh in-memory archive never given a path
    // has nowhere to save to without one being specified here).
    void save(const std::string& path = "");

    MTL::BinaryArchive* mtl() const { return archive_; }

private:
    MTL::BinaryArchive* archive_;  // +1 owned
    std::string         path_;
};

} // namespace mtlpy
