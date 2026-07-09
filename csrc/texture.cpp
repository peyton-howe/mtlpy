#include "texture.h"
#include <stdexcept>

namespace mtlpy {

namespace {

MTL::TextureType texture_type_for(uint32_t dims) {
    switch (dims) {
        case 1: return MTL::TextureType1D;
        case 2: return MTL::TextureType2D;
        case 3: return MTL::TextureType3D;
        default:
            throw std::runtime_error("Texture dims must be 1, 2, or 3 (got " + std::to_string(dims) + ")");
    }
}

MTL::Region region_for(uint32_t dims, uint32_t width, uint32_t height, uint32_t depth) {
    switch (dims) {
        case 1:  return MTL::Region::Make1D(0, width);
        case 2:  return MTL::Region::Make2D(0, 0, width, height);
        default: return MTL::Region::Make3D(0, 0, 0, width, height, depth);
    }
}

} // namespace

Texture::Texture(MTL::Device* device, uint32_t dims, uint32_t pixel_format,
                  uint32_t width, uint32_t height, uint32_t depth)
    : dims_(dims)
{
    // Validate before allocating anything: texture_type_for() throws on an
    // invalid dims, and doing that after MTL::TextureDescriptor::alloc()
    // would leak the descriptor on the way out.
    MTL::TextureType type = texture_type_for(dims);

    auto* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(type);
    desc->setPixelFormat(static_cast<MTL::PixelFormat>(pixel_format));
    desc->setWidth(width);
    desc->setHeight(dims >= 2 ? height : 1);
    desc->setDepth(dims >= 3 ? depth : 1);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    tex_ = device->newTexture(desc);
    desc->release();

    if (!tex_)
        throw std::runtime_error("Failed to allocate Metal texture");
}

Texture::~Texture() {
    tex_->release();
}

void Texture::upload(const void* bytes, size_t bytes_per_row, size_t bytes_per_image) {
    MTL::Region region = region_for(dims_, width(), height(), depth());
    tex_->replaceRegion(region, /*level=*/0, /*slice=*/0, bytes, bytes_per_row, bytes_per_image);
}

void Texture::download(void* bytes, size_t bytes_per_row, size_t bytes_per_image) const {
    MTL::Region region = region_for(dims_, width(), height(), depth());
    tex_->getBytes(bytes, bytes_per_row, bytes_per_image, region, /*level=*/0, /*slice=*/0);
}

uint32_t Texture::width()  const { return (uint32_t)tex_->width(); }
uint32_t Texture::height() const { return (uint32_t)tex_->height(); }
uint32_t Texture::depth()  const { return (uint32_t)tex_->depth(); }

} // namespace mtlpy
