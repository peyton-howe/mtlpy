#pragma once
#include <cstdint>
#include <memory>

namespace mtlpy {

class Device;
class Event;
class Texture;
class Surface;

// One acquired CAMetalLayer drawable, for zero-copy rendering: kernels write
// straight into its texture, then Surface::present_frame() presents it --
// no intermediate texture, no copy pass. Holding a Frame holds one of the
// layer's max_drawables drawables, so present (or drop) it promptly.
class Frame {
public:
    ~Frame();
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;

    // A new owning Texture wrapper around the drawable's texture (its own
    // retain). Only valid to write until the Frame is presented -- after
    // that Core Animation owns the contents and recycles the texture.
    Texture* texture() const;
    uint32_t width() const;
    uint32_t height() const;
    uint32_t pixel_format() const;
    bool presented() const;

private:
    friend class Surface;
    struct Impl;  // holds the id<CAMetalDrawable>; defined in surface.mm
    Frame(const Surface* owner, std::unique_ptr<Impl> impl);
    const Surface*        owner_;
    std::unique_ptr<Impl> impl_;
};

// Displays mtlpy Textures in a native window via a CAMetalLayer -- e.g. a
// PySide6 QWidget (see examples/qt/metal_widget.py), whose winId() is its NSView*.
//
// The layer is added as a *sublayer* of the view's own backing layer rather
// than replacing it, so toolkits like Qt that manage their NSView's backing
// layer themselves keep working unchanged.
//
// Two ways to display:
//
// - Zero-copy: next_frame() hands out the next drawable as a Frame; kernels
//   write directly into Frame::texture(), and present_frame() presents it.
//   The drawable's texture *is* what gets composited -- nothing is copied.
//   The texture is in the drawable's pixel format (bgra8Unorm by default;
//   kernels still write float4 RGBA, Metal handles the BGRA swizzle) and
//   sized to the view's pixels by default.
//
// - present(texture): draws an existing texture into the next drawable --
//   one GPU pass, for textures that must persist across frames or are in a
//   format CAMetalLayer can't display directly.
//
// present() draws the source texture into the next drawable with a
// tiny fullscreen-triangle render pass (not a blit), so the source can be
// any float-readable 2D format (rgba8Unorm, r32Float, rgba16Float, ...)
// regardless of the drawable's pixel format -- CAMetalLayer only supports
// a handful (BGRA8, RGBA16Float, 10-bit), none of which are the usual
// rgba8Unorm compute output. The drawable is sized to the source texture
// and Core Animation scales it to the view (see the gravity param), so the
// texture never has to track the window's pixel size.
//
// Threading: construction, update_layout() and close() touch AppKit and
// must run on the main thread. present() only touches the layer and the
// GPU, so it can run from any thread (it's bound with the GIL released).
class Surface {
public:
    enum Gravity : uint32_t { GravityAspect = 0, GravityFill = 1, GravityStretch = 2, GravityCenter = 3 };
    enum Colorspace : uint32_t {
        ColorspaceNone = 0,                // leave untagged (no color matching)
        ColorspaceSRGB = 1,
        ColorspaceDisplayP3 = 2,
        ColorspaceLinearSRGB = 3,
        ColorspaceExtendedLinearSRGB = 4,  // for rgba16Float + edr
    };

    // view_ptr is an NSView* (e.g. int(QWidget.winId()) on macOS).
    // pixel_format is a raw MTL::PixelFormat supported by CAMetalLayer
    // (BGRA8Unorm, BGRA8Unorm_sRGB, RGBA16Float, RGB10A2Unorm, BGR10A2Unorm).
    Surface(Device* device, uintptr_t view_ptr, uint32_t pixel_format,
            uint32_t gravity, bool linear_filter, uint32_t colorspace, bool edr,
            bool display_sync, uint32_t max_drawables);
    ~Surface();

    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    // Draws src into the next drawable and presents it, on the Device's own
    // queue (so it's ordered after any dispatch already committed there).
    // wait_event/wait_value additionally make the GPU wait for that event
    // first -- needed when src was produced on a secondary Queue.
    // Returns false (frame skipped) if no drawable became available within
    // CAMetalLayer's 1s timeout, e.g. while the window is occluded.
    bool present(Texture* src, bool wait, Event* wait_event, uint64_t wait_value);

    // Zero-copy: acquires the next drawable, sized width x height (0 x 0 =
    // the view's current size in pixels, as of the last update_layout()).
    // Returns nullptr if no drawable became available within the 1s timeout
    // or the view has no area (hidden/minimized). The first call switches
    // the layer's framebufferOnly off so drawables are shader-writable.
    Frame* next_frame(uint32_t width, uint32_t height);

    // Presents a Frame from this Surface's next_frame() after any GPU work
    // already committed to the Device's default queue (i.e. the kernels
    // that wrote it). wait/wait_event/wait_value as in present().
    void present_frame(Frame* frame, bool wait, Event* wait_event, uint64_t wait_value);

    // Re-syncs the layer's frame to the view's bounds. Call from the view's
    // resize handler (main thread).
    void update_layout();

    // Detaches the layer from the view and frees GPU state. Main thread.
    // Idempotent; present() afterwards throws.
    void close();
    bool closed() const;

    // The view's size in pixels as of the last update_layout() (or
    // construction) -- next_frame()'s default size.
    uint32_t view_width() const;
    uint32_t view_height() const;

    // GPU execution time (GPUEndTime - GPUStartTime, seconds) of the most
    // recently *completed* present -- the copy pass for present(texture),
    // ~0 for a zero-copy present_frame(). Updated asynchronously from the
    // command buffer's completion handler; 0 before any present completes.
    double last_present_gpu_time() const;

    uint32_t drawable_width() const;
    uint32_t drawable_height() const;
    uint32_t pixel_format() const { return pixel_format_; }

private:
    struct Impl;  // holds the ObjC objects; defined in surface.mm
    std::unique_ptr<Impl> impl_;
    Device*  device_;
    uint32_t pixel_format_;
};

} // namespace mtlpy
