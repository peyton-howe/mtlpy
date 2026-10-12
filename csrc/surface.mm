// Compiled with -fobjc-arc (see CMakeLists.txt) -- the only ARC translation
// unit in the project, so the ObjC objects in Surface::Impl are managed by
// ARC instead of the manual retain/release the metal-cpp files use.
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include "surface.h"
#include "device.h"
#include "event.h"
#include "texture.h"
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

#if !__has_feature(objc_arc)
#error "surface.mm must be compiled with -fobjc-arc"
#endif

namespace mtlpy {

namespace {

// Draws the source texture 1:1 into the drawable (which present() sizes to
// match it). read() rather than sample() since the sizes are identical --
// exact texels, and no sampler. mode widens 1- and 2-channel sources to
// an opaque RGB image (r -> gray, rg -> rg0) instead of red-only output.
const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 pos [[position]]; };

vertex VOut mtlpy_surface_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

fragment float4 mtlpy_surface_fs(VOut in [[stage_in]],
                                 texture2d<float, access::read> src [[texture(0)]],
                                 constant uint& mode [[buffer(0)]]) {
    float4 c = src.read(uint2(in.pos.xy));
    if (mode == 1) return float4(c.rrr, 1.0);
    if (mode == 2) return float4(c.rg, 0.0, 1.0);
    return float4(c.rgb, 1.0);
}
)";

enum : uint32_t { ModeRGBA = 0, ModeGray = 1, ModeRG = 2 };

bool is_drawable_format(uint32_t f) {
    switch ((MTLPixelFormat)f) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGBA16Float:
        case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatBGR10A2Unorm:
            return true;
        default:
            return false;
    }
}

// Integer and depth/stencil formats can't be bound as texture2d<float>.
bool is_float_readable(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatR8Uint:      case MTLPixelFormatR8Sint:
        case MTLPixelFormatR16Uint:     case MTLPixelFormatR16Sint:
        case MTLPixelFormatRG8Uint:     case MTLPixelFormatRG8Sint:
        case MTLPixelFormatR32Uint:     case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG16Uint:    case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRGBA8Uint:   case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGB10A2Uint:
        case MTLPixelFormatRG32Uint:    case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA16Uint:  case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatRGBA32Uint:  case MTLPixelFormatRGBA32Sint:
        case MTLPixelFormatDepth16Unorm:
        case MTLPixelFormatDepth32Float:
        case MTLPixelFormatStencil8:
        case MTLPixelFormatDepth32Float_Stencil8:
        case MTLPixelFormatX32_Stencil8:
            return false;
        default:
            return true;
    }
}

uint32_t mode_for(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatR8Unorm:  case MTLPixelFormatR8Unorm_sRGB:
        case MTLPixelFormatR8Snorm:  case MTLPixelFormatR16Unorm:
        case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Float:
        case MTLPixelFormatR32Float:
            return ModeGray;
        case MTLPixelFormatRG8Unorm:  case MTLPixelFormatRG8Unorm_sRGB:
        case MTLPixelFormatRG8Snorm:  case MTLPixelFormatRG16Unorm:
        case MTLPixelFormatRG16Snorm: case MTLPixelFormatRG16Float:
        case MTLPixelFormatRG32Float:
            return ModeRG;
        default:
            return ModeRGBA;
    }
}

NSString* gravity_for(uint32_t g) {
    switch (g) {
        case Surface::GravityAspect:  return kCAGravityResizeAspect;
        case Surface::GravityFill:    return kCAGravityResizeAspectFill;
        case Surface::GravityStretch: return kCAGravityResize;
        case Surface::GravityCenter:  return kCAGravityCenter;
        default: throw std::invalid_argument("Surface: unknown gravity " + std::to_string(g));
    }
}

CGColorSpaceRef create_colorspace(uint32_t c) {
    switch (c) {
        case Surface::ColorspaceNone:               return nullptr;
        case Surface::ColorspaceSRGB:               return CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        case Surface::ColorspaceDisplayP3:          return CGColorSpaceCreateWithName(kCGColorSpaceDisplayP3);
        case Surface::ColorspaceLinearSRGB:         return CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB);
        case Surface::ColorspaceExtendedLinearSRGB: return CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
        default: throw std::invalid_argument("Surface: unknown colorspace " + std::to_string(c));
    }
}

// CALayer hierarchy changes belong on the main thread. close()/~Surface()
// can run from anywhere (e.g. Python GC on a worker thread), so hop over
// when needed -- the block keeps the layer alive until it runs.
void detach_layer(CAMetalLayer* layer) {
    if (!layer) return;
    if ([NSThread isMainThread]) {
        [layer removeFromSuperlayer];
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{ [layer removeFromSuperlayer]; });
    }
}

std::string ns_error_message(NSError* err) {
    return err ? std::string(err.localizedDescription.UTF8String) : std::string("unknown error");
}

void commit(id<MTLCommandBuffer> cmd, bool wait, const char* what,
            std::shared_ptr<std::atomic<double>> gpu_time, Event* wait_event) {
    // Encoded event waits do not retain the event. Keep a strong reference
    // in the completion block even when the Python caller drops its event.
    __block id<MTLEvent> retained_event = wait_event ? (__bridge id<MTLEvent>)wait_event->mtl() : nil;
    // shared_ptr, not a pointer into Surface: the handler can run after
    // the Surface is gone.
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        gpu_time->store(cb.GPUEndTime - cb.GPUStartTime, std::memory_order_relaxed);
        retained_event = nil;
    }];
    [cmd commit];
    if (wait) {
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError)
            throw std::runtime_error(std::string(what) + ": GPU error: " + ns_error_message(cmd.error));
        // Completion handlers may still be pending when the wait returns.
        gpu_time->store(cmd.GPUEndTime - cmd.GPUStartTime, std::memory_order_relaxed);
    }
}

// The view's size in backing pixels -- what a 1:1 zero-copy frame should be.
// convertSizeToBacking: uses the window's screen (or the main screen before
// the view is in a window), unlike layer.contentsScale, which a toolkit may
// not have synced yet.
CGSize view_pixel_size(NSView* view) {
    return [view convertSizeToBacking:view.bounds.size];
}

} // namespace

struct Surface::Impl {
    // Lock before mutex; close() never waits for drawable acquisition.
    std::mutex                acquisition_mutex;
    std::mutex                mutex;
    CAMetalLayer*              layer = nil;
    // Weak: the toolkit (e.g. Qt) owns the view and may destroy it before
    // this Surface -- we only need it again for update_layout().
    __weak NSView*             view = nil;
    id<MTLRenderPipelineState> pipeline = nil;
    bool                       closed = false;
    // Written on the main thread (update_layout), read by next_frame() from
    // any thread.
    std::atomic<uint32_t>      view_w{0}, view_h{0};
    std::shared_ptr<std::atomic<double>> gpu_time = std::make_shared<std::atomic<double>>(0.0);

    void record_view_size(NSView* v) {
        CGSize px = view_pixel_size(v);
        view_w = (uint32_t)px.width;
        view_h = (uint32_t)px.height;
    }
};

struct Frame::Impl {
    id<CAMetalDrawable> drawable = nil;
    bool                presented = false;
};

Frame::Frame(const Surface* owner, std::unique_ptr<Impl> impl)
    : owner_(owner), impl_(std::move(impl)) {}

Frame::~Frame() {
    @autoreleasepool { impl_.reset(); }  // returns the drawable to the layer's pool
}

Texture* Frame::texture() const {
    if (impl_->presented)
        throw std::runtime_error("Frame.texture: frame was already presented");
    id<MTLTexture> tex = impl_->drawable.texture;
    // +1 for Texture, which releases it in its destructor.
    return new Texture((__bridge_retained MTL::Texture*)tex, 2,
                       tex.storageMode != MTLStorageModeShared);
}

uint32_t Frame::width() const        { return (uint32_t)impl_->drawable.texture.width; }
uint32_t Frame::height() const       { return (uint32_t)impl_->drawable.texture.height; }
uint32_t Frame::pixel_format() const { return (uint32_t)impl_->drawable.texture.pixelFormat; }
bool     Frame::presented() const    { return impl_->presented; }

Surface::Surface(Device* device, uintptr_t view_ptr, uint32_t pixel_format,
                 uint32_t gravity, bool linear_filter, uint32_t colorspace, bool edr,
                 bool display_sync, uint32_t max_drawables)
    : impl_(std::make_unique<Impl>()), device_(device), pixel_format_(pixel_format)
{
    @autoreleasepool {
        if (!view_ptr)
            throw std::invalid_argument("Surface: null view_ptr");
        if (![NSThread isMainThread])
            throw std::runtime_error("Surface: must be created on the main thread");
        if (!is_drawable_format(pixel_format))
            throw std::invalid_argument(
                "Surface: pixel_format must be one CAMetalLayer supports "
                "(bgra8Unorm, bgra8Unorm_srgb, rgba16Float, rgb10a2Unorm, bgr10a2Unorm)");
        if (max_drawables != 2 && max_drawables != 3)
            throw std::invalid_argument("Surface: max_drawables must be 2 or 3");

        // No way to validate an arbitrary integer as an object pointer
        // without dereferencing it -- a garbage view_ptr crashes here.
        NSView* view = (__bridge NSView*)(void*)view_ptr;
        if (![view isKindOfClass:[NSView class]])
            throw std::invalid_argument("Surface: view_ptr is not an NSView");

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device->mtl();

        NSError* err = nil;
        id<MTLLibrary> lib = [mtl_device newLibraryWithSource:@(kShaderSource) options:nil error:&err];
        if (!lib)
            throw std::runtime_error("Surface: shader compile failed: " + ns_error_message(err));
        MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction   = [lib newFunctionWithName:@"mtlpy_surface_vs"];
        pd.fragmentFunction = [lib newFunctionWithName:@"mtlpy_surface_fs"];
        pd.colorAttachments[0].pixelFormat = (MTLPixelFormat)pixel_format;
        impl_->pipeline = [mtl_device newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!impl_->pipeline)
            throw std::runtime_error("Surface: pipeline creation failed: " + ns_error_message(err));

        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.device             = mtl_device;
        layer.pixelFormat        = (MTLPixelFormat)pixel_format;
        layer.framebufferOnly    = YES;  // render-pass target only -- lets Metal optimize it
        layer.opaque             = YES;
        layer.displaySyncEnabled = display_sync;
        layer.maximumDrawableCount = max_drawables;
        layer.wantsExtendedDynamicRangeContent = edr;
        layer.contentsGravity    = gravity_for(gravity);
        NSString* filter = linear_filter ? kCAFilterLinear : kCAFilterNearest;
        layer.magnificationFilter = filter;
        layer.minificationFilter  = filter;
        layer.backgroundColor    = CGColorGetConstantColor(kCGColorBlack);  // letterbox bars
        if (CGColorSpaceRef cs = create_colorspace(colorspace)) {
            layer.colorspace = cs;
            CGColorSpaceRelease(cs);
        }
        // No implicit animations when the frame/contents change on resize.
        NSNull* none = [NSNull null];
        layer.actions = @{@"bounds": none, @"position": none, @"contents": none,
                          @"contentsGravity": none};
        layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;

        // Sublayer of the view's own backing layer, not a replacement for
        // it: Qt (and other toolkits) manage their NSView's backing layer
        // and can reset it out from under a replaced one.
        view.wantsLayer = YES;
        CALayer* host = view.layer;
        if (!host)
            throw std::runtime_error("Surface: view has no backing layer");
        layer.frame = host.bounds;
        layer.contentsScale = [view convertSizeToBacking:NSMakeSize(1, 1)].width;
        [host addSublayer:layer];

        impl_->layer = layer;
        impl_->view  = view;
        impl_->record_view_size(view);
    }
}

Surface::~Surface() {
    @autoreleasepool {
        detach_layer(impl_->layer);
        impl_.reset();
    }
}

void Surface::close() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) return;
    @autoreleasepool {
        detach_layer(impl_->layer);
        impl_->layer    = nil;
        impl_->pipeline = nil;
        impl_->view     = nil;
        impl_->closed   = true;
    }
}

bool Surface::closed() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->closed;
}

void Surface::update_layout() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) return;
    if (![NSThread isMainThread])
        throw std::runtime_error("Surface.update_layout: must be called on the main thread");
    @autoreleasepool {
        NSView* view = impl_->view;
        if (!view || !view.layer) return;
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        impl_->layer.frame = view.layer.bounds;
        if (view.window)
            impl_->layer.contentsScale = view.window.backingScaleFactor;
        [CATransaction commit];
        impl_->record_view_size(view);
    }
}

double Surface::last_present_gpu_time() const {
    return impl_->gpu_time->load(std::memory_order_relaxed);
}

uint32_t Surface::view_width() const  { return impl_->view_w; }
uint32_t Surface::view_height() const { return impl_->view_h; }

uint32_t Surface::drawable_width() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->layer ? (uint32_t)impl_->layer.drawableSize.width : 0;
}

uint32_t Surface::drawable_height() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->layer ? (uint32_t)impl_->layer.drawableSize.height : 0;
}

bool Surface::present(Texture* src, bool wait, Event* wait_event, uint64_t wait_value) {
    // Snapshot strong references so close() can tear down without waiting
    // for nextDrawable or GPU completion on a render thread.
    CAMetalLayer* layer;
    id<MTLRenderPipelineState> pipeline;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->closed)
            throw std::runtime_error("Surface.present: surface is closed");
        layer = impl_->layer;
        pipeline = impl_->pipeline;
    }
    if (!src)
        throw std::invalid_argument("Surface.present: null texture");

    @autoreleasepool {
        id<MTLTexture> tex = (__bridge id<MTLTexture>)src->mtl();
        if (tex.device != (__bridge id<MTLDevice>)device_->mtl())
            throw std::invalid_argument("Surface.present: texture belongs to a different Device");
        if (tex.textureType != MTLTextureType2D)
            throw std::invalid_argument("Surface.present: texture must be 2D");
        if (!(tex.usage & MTLTextureUsageShaderRead))
            throw std::invalid_argument("Surface.present: texture must be readable (created with readable=True)");
        if (!is_float_readable(tex.pixelFormat))
            throw std::invalid_argument(
                "Surface.present: integer textures can't be displayed directly -- "
                "convert to a normalized or float format first");

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)device_->queue();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd)
            throw std::runtime_error("Surface.present: failed to create command buffer");
        if (wait_event)
            [cmd encodeWaitForEvent:(__bridge id<MTLEvent>)wait_event->mtl() value:wait_value];

        // Acquire the drawable as late as possible -- it's a scarce resource
        // (max_drawables of them) and this can block until one frees up.
        id<CAMetalDrawable> drawable;
        {
            std::lock_guard<std::mutex> acquisition_lock(impl_->acquisition_mutex);
            CGSize size = CGSizeMake(tex.width, tex.height);
            if (!CGSizeEqualToSize(layer.drawableSize, size))
                layer.drawableSize = size;
            drawable = [layer nextDrawable];
        }
        if (!drawable)
            return false;

        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture     = drawable.texture;
        rp.colorAttachments[0].loadAction  = MTLLoadActionDontCare;  // every pixel is overwritten
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;

        uint32_t mode = mode_for(tex.pixelFormat);
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:pipeline];
        [enc setFragmentTexture:tex atIndex:0];
        [enc setFragmentBytes:&mode length:sizeof(mode) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];

        [cmd presentDrawable:drawable];
        commit(cmd, wait, "Surface.present", impl_->gpu_time, wait_event);
        return true;
    }
}

Frame* Surface::next_frame(uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> acquisition_lock(impl_->acquisition_mutex);
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->closed)
        throw std::runtime_error("Surface.next_frame: surface is closed");
    if ((width == 0) != (height == 0))
        throw std::invalid_argument("Surface.next_frame: give both width and height, or neither");
    if (width == 0) {
        // On the main thread (the usual case, e.g. a Qt timer) the view can
        // be read directly, so don't rely on update_layout() having run.
        if ([NSThread isMainThread]) {
            @autoreleasepool {
                if (NSView* view = impl_->view)
                    impl_->record_view_size(view);
            }
        }
        width  = impl_->view_w;
        height = impl_->view_h;
        if (width == 0 || height == 0)
            return nullptr;  // no area to draw into (hidden/minimized/not laid out yet)
    }

    @autoreleasepool {
        CAMetalLayer* layer = impl_->layer;
        // Drawables are render-target-only until this is off. Flipped
        // lazily so present()-only users keep the optimized drawables.
        if (layer.framebufferOnly)
            layer.framebufferOnly = NO;
        CGSize size = CGSizeMake(width, height);
        if (!CGSizeEqualToSize(layer.drawableSize, size))
            layer.drawableSize = size;

        lock.unlock();  // layer stays alive if close() runs while acquiring
        id<CAMetalDrawable> drawable = [layer nextDrawable];
        if (!drawable)
            return nullptr;
        auto impl = std::make_unique<Frame::Impl>();
        impl->drawable = drawable;  // strong: outlives this pool
        return new Frame(this, std::move(impl));
    }
}

void Surface::present_frame(Frame* frame, bool wait, Event* wait_event, uint64_t wait_value) {
    if (closed())
        throw std::runtime_error("Surface.present: surface is closed");
    if (!frame)
        throw std::invalid_argument("Surface.present: null frame");
    if (frame->owner_ != this)
        throw std::invalid_argument("Surface.present: frame came from a different Surface");
    if (frame->impl_->presented)
        throw std::runtime_error("Surface.present: frame was already presented");

    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)device_->queue();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd)
            throw std::runtime_error("Surface.present: failed to create command buffer");
        if (wait_event)
            [cmd encodeWaitForEvent:(__bridge id<MTLEvent>)wait_event->mtl() value:wait_value];
        [cmd presentDrawable:frame->impl_->drawable];
        frame->impl_->presented = true;
        commit(cmd, wait, "Surface.present", impl_->gpu_time, wait_event);
    }
}

} // namespace mtlpy
