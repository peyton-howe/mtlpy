#include "binary_archive.h"
#include "pool_guard.h"
#include <filesystem>
#include <stdexcept>

namespace mtlpy {

namespace {

namespace fs = std::filesystem;

NS::URL* url_for(const std::string& path) {
    return NS::URL::fileURLWithPath(
        NS::String::string(path.c_str(), NS::UTF8StringEncoding));
}

} // namespace

BinaryArchive::BinaryArchive(MTL::Device* device, const std::string& path)
    : path_(path)
{
    PoolGuard guard;
    auto* descriptor = MTL::BinaryArchiveDescriptor::alloc()->init();
    if (!path.empty() && fs::exists(path))
        descriptor->setUrl(url_for(path));

    NS::Error* error = nullptr;
    archive_ = device->newBinaryArchive(descriptor, &error);
    descriptor->release();
    if (!archive_)
        throw std::runtime_error(
            std::string("Failed to create/open Metal binary archive: ") +
            (error ? error->localizedDescription()->utf8String() : "unknown error"));
}

BinaryArchive::~BinaryArchive() {
    PoolGuard guard;
    archive_->release();
}

void BinaryArchive::save(const std::string& path) {
    const std::string& target = !path.empty() ? path : path_;
    if (target.empty())
        throw std::runtime_error(
            "BinaryArchive has no path to save to -- pass one explicitly to save()");

    PoolGuard guard;
    std::error_code ec;
    fs::create_directories(fs::path(target).parent_path(), ec);

    NS::Error* error = nullptr;
    bool ok = archive_->serializeToURL(url_for(target), &error);
    if (!ok)
        throw std::runtime_error(
            std::string("Failed to save Metal binary archive: ") +
            (error ? error->localizedDescription()->utf8String() : "unknown error"));
}

} // namespace mtlpy
