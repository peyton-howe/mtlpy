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
class CommandBuffer;
class Heap;
class Queue;
class Event;
class SharedEvent;
class SharedEventHandle;
class Fence;

class Device {
public:
    // index < 0 (the default) uses CreateSystemDefaultDevice(); index >= 0
    // selects that position in available_device_names()/CopyAllDevices(),
    // for multi-GPU machines.
    explicit Device(int index = -1);
    ~Device();

    // storage_mode -- see Buffer's constructor in buffer.h.
    Buffer*   create_buffer(size_t size_bytes, uint32_t storage_mode);

    // storage_mode -- see Heap's constructor in heap.h.
    Heap*     create_heap(size_t size_bytes, uint32_t storage_mode);

    Pipeline* compile(const std::string& source, const std::string& function_name);

    // dims is 1/2/3 (see Texture); pixel_format is a raw MTL::PixelFormat
    // value (see src/mtlpy/utils.py's pixel format table). usage/
    // private_storage -- see Texture's constructor in texture.h.
    Texture* create_texture(uint32_t dims, uint32_t pixel_format,
                             uint32_t width, uint32_t height, uint32_t depth,
                             uint32_t usage, bool private_storage);

    // Hardware-blit upload: copies buf's memory into tex via
    // MTLBlitCommandEncoder rather than Texture::upload()'s CPU-side
    // replaceRegion. tex keeps its normal (possibly tiled/swizzled)
    // internal layout -- the blit engine does the retiling on the GPU side,
    // concurrently with the CPU, instead of the CPU computing it inline.
    // bytes_per_row/bytes_per_image describe buf's layout starting at
    // offset (src/mtlpy/texture.py computes these as tightly packed, same
    // convention as Texture::upload's bytes_per_row).
    void blit_upload_texture(Buffer* buf, size_t offset, Texture* tex,
                              size_t bytes_per_row, size_t bytes_per_image, bool wait);

    // The read counterpart to blit_upload_texture(): hardware-blit copy of
    // tex's pixel data into buf (MTLBlitCommandEncoder::copyFromTexture,
    // texture-to-buffer overload) starting at offset, instead of
    // Texture::download()'s CPU-side getBytes(). Moves raw bytes with no
    // shader/format-conversion pass, so -- like copy_texture() below --
    // this works for any pixel format (Unorm included, unlike
    // buffer_from_texture()'s compute-kernel readback) and any combination
    // of Shared/Private storage on either side. bytes_per_row/
    // bytes_per_image describe buf's layout starting at offset, same
    // convention as blit_upload_texture's.
    void blit_download_texture(Texture* tex, Buffer* buf, size_t offset,
                                size_t bytes_per_row, size_t bytes_per_image, bool wait);

    // Encodes MTLBlitCommandEncoder::optimizeContentsForGPUAccess -- lets
    // Metal repack a texture's contents into its preferred GPU-side layout
    // after the fact. Private-storage textures already get this for free at
    // creation (per Apple's docs); this exists for the Shared-storage case,
    // which doesn't. tex's contents must already be populated (upload()/
    // upload_from_buffer()) before calling this.
    void optimize_texture_for_gpu_access(Texture* tex, bool wait);

    // Hardware-blit texture-to-texture copy (MTLBlitCommandEncoder::
    // copyFromTexture, whole-texture overload): src and dst must already
    // match in pixel format and dimensions -- this moves raw bytes, no
    // shader/format-conversion path, so it works for any pixel format
    // (including Unorm, unlike Texture::to_buffer()) and any combination of
    // Shared/Private storage on either side.
    void copy_texture(Texture* src, Texture* dst, bool wait);

    // Hardware-blit buffer-to-buffer copy (MTLBlitCommandEncoder::
    // copyFromBuffer) -- the Buffer counterpart to copy_texture(), and the
    // mechanism Buffer.to_storage() (src/mtlpy/buffer.py) uses to
    // materialize a CPU-readable Shared copy of a Private/Managed Buffer.
    // Works for any combination of storage modes on either side, same as
    // copy_texture().
    void copy_buffer(Buffer* src, size_t src_offset, Buffer* dst, size_t dst_offset,
                      size_t size_bytes, bool wait);

    Sampler* create_sampler(bool linear, bool repeat);

    // Lets multiple Pipeline::run() dispatches batch into one command-
    // buffer submission -- see CommandBuffer's own doc comment. queue
    // (default nullptr) targets a secondary Queue (see create_queue())
    // instead of this Device's own default queue -- the mechanism for
    // running work on more than one MTL::CommandQueue.
    CommandBuffer* create_command_buffer(Queue* queue = nullptr);

    // A second MTL::CommandQueue beyond this Device's own default one --
    // see Queue's class doc comment.
    Queue* create_queue();

    // GPU-side-only cross-command-buffer/cross-queue signal -- see Event.
    Event* create_event();

    // Like create_event(), but adds CPU-visible signal/wait and a
    // cross-process-exportable handle -- see SharedEvent.
    SharedEvent* create_shared_event();
    SharedEvent* create_shared_event_from_handle(SharedEventHandle* handle);

    // Same-queue producer/consumer ordering primitive -- see Fence.
    Fence* create_fence();

    uint32_t max_threads_per_threadgroup() const;
    void     flush_cache();

    static std::vector<std::string> available_device_names();

    // The id<MTLDevice> handle itself (see Buffer::mtl()/Texture::mtl() for
    // the same convention) -- exposed so external code (another native
    // library, a hand-written PyObjC/Metal bridge, ...) can confirm a
    // Buffer/Texture it was handed belongs to *this* device before doing
    // anything with it. Metal forbids referencing a resource from a
    // different MTLDevice in the same command encoder -- there's no
    // cross-device fallback, so this is a hard precondition, not an
    // optimization.
    MTL::Device* mtl() const { return device_; }

private:
    MTL::Device*       device_;
    MTL::CommandQueue* queue_;
    PipelineCache*     cache_;
};

} // namespace mtlpy
