#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <vector>
#include "device.h"
#include "binary_archive.h"
#include "buffer.h"
#include "capture.h"
#include "command_buffer.h"
#include "dlpack.h"
#include "event.h"
#include "fence.h"
#include "heap.h"
#include "pipeline.h"
#include "queue.h"
#include "sampler.h"
#include "texture.h"

namespace nb = nanobind;
using namespace mtlpy;

NB_MODULE(_mtlpy, m) {
    m.doc() = "Apple Metal GPU compute bindings";

    m.def("list_devices", &Device::available_device_names);

    nb::class_<Device>(m, "Device")
        .def(nb::init<int, std::optional<std::string>>(),
             nb::arg("index") = -1, nb::arg("cache_path") = std::nullopt)
        .def("create_buffer", &Device::create_buffer,
             nb::arg("size_bytes"), nb::arg("storage_mode"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Buffer is alive
        .def("compile", &Device::compile,
             nb::arg("source"), nb::arg("function_name"), nb::arg("archive") = nullptr,
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Pipeline is alive
        .def("max_threads_per_threadgroup", &Device::max_threads_per_threadgroup)
        .def("flush_cache", &Device::flush_cache, nb::arg("path") = std::nullopt)
        .def("create_binary_archive", &Device::create_binary_archive,
             nb::arg("path"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while BinaryArchive is alive
        .def_prop_ro("pipeline_cache_size", &Device::pipeline_cache_size)
        .def_prop_ro("pipeline_cache_path", &Device::pipeline_cache_path)
        .def("start_capture", &Device::start_capture, nb::arg("path") = std::nullopt)
        .def("stop_capture", &Device::stop_capture)
        // is_capturing() is a static method (MTLCaptureManager is a
        // process-wide singleton, not per-Device) -- def_static keeps it
        // callable both as Device.is_capturing() and off an instance
        // (self._dev.is_capturing()), the same as a Python staticmethod.
        .def_static("is_capturing", &Device::is_capturing)
        .def("create_capture_scope", &Device::create_capture_scope,
             nb::arg("label"), nb::arg("queue") = nullptr,
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>(),   // keep Device alive while CaptureScope is alive
             nb::keep_alive<0, 3>())   // ...and queue too, if given (same MTLResource-vs-not
                                       // rationale as CommandBuffer::encode_wait_for_event's
                                       // keep_alive, see above)
        .def_prop_ro("mtl_ptr", [](const Device& d) {
            // The id<MTLDevice> handle itself -- raw, non-owning (this
            // Device's destructor still owns the real release()). For
            // handing to external native code (e.g. setting a CAMetalLayer's
            // .device to match, so a Texture from this Device can be blitted
            // into its drawable) -- see Buffer.mtl_ptr's docstring in
            // buffer.py for the same lifetime caveat: valid only as long as
            // this Device object is kept alive Python-side.
            return reinterpret_cast<uintptr_t>(d.mtl());
        })
        .def("create_texture", &Device::create_texture,
             nb::arg("dims"), nb::arg("pixel_format"),
             nb::arg("width"), nb::arg("height"), nb::arg("depth"),
             nb::arg("usage"), nb::arg("private_storage"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Texture is alive
        .def("blit_upload_texture", &Device::blit_upload_texture,
             nb::arg("buf"), nb::arg("offset"), nb::arg("tex"),
             nb::arg("bytes_per_row"), nb::arg("bytes_per_image"), nb::arg("wait"),
             // Same rationale as Pipeline::run's GIL release above -- this
             // blocks on waitUntilCompleted() when wait=True.
             nb::call_guard<nb::gil_scoped_release>())
        .def("blit_download_texture", &Device::blit_download_texture,
             nb::arg("tex"), nb::arg("buf"), nb::arg("offset"),
             nb::arg("bytes_per_row"), nb::arg("bytes_per_image"), nb::arg("wait"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("optimize_texture_for_gpu_access", &Device::optimize_texture_for_gpu_access,
             nb::arg("tex"), nb::arg("wait"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("copy_texture", &Device::copy_texture,
             nb::arg("src"), nb::arg("dst"), nb::arg("wait"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("copy_buffer", &Device::copy_buffer,
             nb::arg("src"), nb::arg("src_offset"), nb::arg("dst"), nb::arg("dst_offset"),
             nb::arg("size_bytes"), nb::arg("wait"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("create_sampler", &Device::create_sampler,
             nb::arg("linear"), nb::arg("repeat"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Sampler is alive
        .def("create_command_buffer", &Device::create_command_buffer,
             nb::arg("queue") = nullptr,
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>(),   // keep Device alive while CommandBuffer is alive
             nb::keep_alive<0, 2>())   // ...and queue too, if given -- the returned
                                       // CommandBuffer's cmd_ is tied to queue's
                                       // MTL::CommandQueue for its whole lifetime,
                                       // same rationale as create_capture_scope's below
        .def("create_heap", &Device::create_heap,
             nb::arg("size_bytes"), nb::arg("storage_mode"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Heap is alive
        .def("create_queue", &Device::create_queue,
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Queue is alive
        .def("create_event", &Device::create_event,
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Event is alive
        .def("create_shared_event", &Device::create_shared_event,
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while SharedEvent is alive
        .def("create_shared_event_from_handle", &Device::create_shared_event_from_handle,
             nb::arg("handle"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while SharedEvent is alive
        .def("create_fence", &Device::create_fence,
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>());  // keep Device alive while Fence is alive

    nb::class_<Buffer>(m, "Buffer")
        .def_prop_ro("data_ptr", [](const Buffer& b) {
            return reinterpret_cast<uintptr_t>(b.contents_ptr());
        })
        .def_prop_ro("mtl_ptr", [](const Buffer& b) {
            // The id<MTLBuffer> handle itself (metal-cpp's MTL::Buffer* IS
            // the id, see csrc/mps/kernels.mm's bridge_device comment), not
            // contents_ptr() -- this is what DLPack's kDLMetal device type
            // expects in DLTensor.data (see _dlpack_capsule below).
            return reinterpret_cast<uintptr_t>(b.mtl());
        })
        .def_prop_ro("size_bytes", &Buffer::size_bytes)
        .def_prop_ro("storage_mode", &Buffer::storage_mode)
        .def("_dlpack_capsule", [](const Buffer& b, uint8_t dtype_code, uint8_t dtype_bits,
                                    std::vector<int64_t> shape) {
            // Backs Buffer.__dlpack__ (src/mtlpy/buffer.py) -- builds a
            // DLManagedTensor tagged kDLMetal, with DLTensor.data set to the
            // id<MTLBuffer> handle itself (verified zero-copy against MLX:
            // mx.asarray(buf, copy=False) reads/writes the same underlying
            // Metal allocation, no copy). Buffer.__dlpack__ already rejects a
            // non-Shared Buffer before ever reaching here (Private has no
            // CPU-visible memory to hand a consumer zero-copy, and Managed
            // needs an explicit synchronize DLPack consumers won't do), so
            // this can assume Shared storage unconditionally.
            //
            // Lifetime: retain()/release() the underlying MTL::Buffer
            // directly (independent of this C++ Buffer wrapper or the
            // Python object owning it) -- that's the only thing that
            // actually needs to outlive the exporting Buffer, and it's
            // exactly what keeps the shared allocation alive for as long as
            // the DLPack consumer holds it.
            // Only the shape array needs its own heap lifetime -- the
            // MTL::Buffer* is recoverable from dl_tensor.data itself (set
            // right below), so the deleter derives it from there instead of
            // duplicating it in a separate field.
            struct Ctx {
                std::vector<int64_t> shape;
            };
            auto* ctx = new Ctx{std::move(shape)};

            auto* tensor = new DLManagedTensor();
            tensor->dl_tensor.data       = reinterpret_cast<void*>(b.mtl()->retain());
            tensor->dl_tensor.device     = DLDevice{kDLMetal, 0};
            tensor->dl_tensor.ndim       = static_cast<int32_t>(ctx->shape.size());
            tensor->dl_tensor.dtype      = DLDataType{dtype_code, dtype_bits, 1};
            tensor->dl_tensor.shape      = ctx->shape.data();
            tensor->dl_tensor.strides    = nullptr;  // always C-contiguous (see buffer.py)
            tensor->dl_tensor.byte_offset = 0;
            tensor->manager_ctx = ctx;
            tensor->deleter = [](DLManagedTensor* self) {
                reinterpret_cast<MTL::Buffer*>(self->dl_tensor.data)->release();
                delete static_cast<Ctx*>(self->manager_ctx);
                delete self;
            };

            // Standard DLPack Python-capsule handoff: a consumer that takes
            // ownership renames the capsule "dltensor" -> "used_dltensor"
            // and calls tensor->deleter itself later (when its own imported
            // array is destroyed). If the capsule is instead dropped without
            // ever being consumed (still named "dltensor"), this destructor
            // must call the deleter itself so the retained MTL::Buffer isn't
            // leaked.
            //
            // nanobind's own nb::capsule only supports a void*-only cleanup
            // callback (no access to the PyObject* capsule itself), which
            // can't express the "was this renamed to used_dltensor?" check
            // below -- so this builds the raw CPython capsule directly
            // (exactly what nb::capsule would do internally anyway) and
            // hands the PyObject* to nb::steal, same as py::capsule did.
            PyObject* capsule = PyCapsule_New(tensor, "dltensor", [](PyObject* capsule) {
                if (PyCapsule_IsValid(capsule, "used_dltensor"))
                    return;
                auto* self = static_cast<DLManagedTensor*>(
                    PyCapsule_GetPointer(capsule, "dltensor"));
                if (self && self->deleter)
                    self->deleter(self);
            });
            if (!capsule) {
                // PyCapsule_New itself failed (allocation failure, or an
                // error already set) -- without this, the retain() just
                // taken above and both heap allocations would leak, since
                // nothing else owns them yet. Reuse the same deleter that
                // would otherwise run when the capsule is GC'd unconsumed.
                tensor->deleter(tensor);
                throw nb::python_error();
            }
            return nb::steal<nb::object>(capsule);
        }, nb::arg("dtype_code"), nb::arg("dtype_bits"), nb::arg("shape"));

    nb::class_<BinaryArchive>(m, "BinaryArchive")
        .def("save", &BinaryArchive::save, nb::arg("path") = "");

    nb::class_<CaptureScope>(m, "CaptureScope")
        .def("begin_scope", &CaptureScope::begin_scope)
        .def("end_scope", &CaptureScope::end_scope);

    nb::class_<Heap>(m, "Heap")
        .def("new_buffer", &Heap::new_buffer,
             nb::arg("size_bytes"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Heap alive while Buffer is alive
        .def("new_texture", &Heap::new_texture,
             nb::arg("dims"), nb::arg("pixel_format"),
             nb::arg("width"), nb::arg("height"), nb::arg("depth"),
             nb::arg("usage"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Heap alive while Texture is alive
        .def_prop_ro("size", &Heap::size)
        .def_prop_ro("used_size", &Heap::used_size)
        .def_prop_ro("storage_mode", &Heap::storage_mode);

    nb::class_<Texture>(m, "Texture")
        .def("upload", [](Texture& t, nb::object data, size_t bytes_per_row, size_t bytes_per_image) {
            // No py::buffer/buffer_info equivalent in nanobind (its answer
            // to buffer-protocol interop is nb::ndarray<>, which negotiates
            // shape/dtype we don't need here -- Texture::upload just wants a
            // raw pointer, treated as bytes_per_row/bytes_per_image-packed
            // data the Python side already computed). Going straight to the
            // CPython buffer protocol C API is simpler than either: this
            // codebase's own upload() (src/mtlpy/texture.py) always passes
            // np.ascontiguousarray(...) first, so PyBUF_SIMPLE (contiguous,
            // no strides/format needed) always succeeds in practice.
            Py_buffer view;
            if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_SIMPLE) != 0)
                throw nb::python_error();
            struct BufGuard {
                Py_buffer* v;
                ~BufGuard() { PyBuffer_Release(v); }
            } guard{&view};
            t.upload(view.buf, bytes_per_row, bytes_per_image);
        }, nb::arg("data"), nb::arg("bytes_per_row"), nb::arg("bytes_per_image"))
        .def("download", [](const Texture& t, size_t nbytes, size_t bytes_per_row, size_t bytes_per_image) {
            // PyBytes_FromStringAndSize(nullptr, n) allocates an n-byte
            // bytes object *without* zero-filling or copying into it --
            // unlike building a std::string (zero-init) and then handing it
            // to nb::bytes (a second full copy), this touches the payload
            // exactly once, via getBytes() writing straight into the
            // PyBytesObject's own storage. Wrap it in nb::bytes immediately
            // so it's exception-safe if t.download() throws.
            PyObject* obj = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)nbytes);
            if (!obj)
                throw nb::python_error();
            nb::bytes result = nb::steal<nb::bytes>(obj);
            t.download(PyBytes_AS_STRING(obj), bytes_per_row, bytes_per_image);
            return result;
        }, nb::arg("nbytes"), nb::arg("bytes_per_row"), nb::arg("bytes_per_image"))
        .def_prop_ro("width",  &Texture::width)
        .def_prop_ro("height", &Texture::height)
        .def_prop_ro("depth",  &Texture::depth)
        .def_prop_ro("dims",   &Texture::dims)
        .def_prop_ro("is_private", &Texture::is_private)
        .def_prop_ro("mtl_ptr", [](const Texture& t) {
            // The id<MTLTexture> handle itself -- raw, non-owning (this
            // Texture's destructor still owns the real release()). Unlike
            // Buffer's DLPack export, there's no automatic cross-library
            // lifetime management here (no retain tied to a capsule
            // deleter) -- see Texture.mtl_ptr's docstring in texture.py for
            // what that means for callers.
            return reinterpret_cast<uintptr_t>(t.mtl());
        });

    nb::class_<Sampler>(m, "Sampler");

    nb::class_<Queue>(m, "Queue")
        .def_prop_ro("mtl_ptr", [](const Queue& q) {
            // Same raw-pointer, non-owning convention as Device.mtl_ptr/
            // Buffer.mtl_ptr -- valid only as long as this Queue is alive.
            return reinterpret_cast<uintptr_t>(q.mtl());
        });

    nb::class_<Fence>(m, "Fence")
        .def_prop_ro("mtl_ptr", [](const Fence& f) {
            return reinterpret_cast<uintptr_t>(f.mtl());
        });

    nb::class_<Event>(m, "Event")
        .def_prop_ro("mtl_ptr", [](const Event& e) {
            return reinterpret_cast<uintptr_t>(e.mtl());
        });

    nb::class_<SharedEventHandle>(m, "SharedEventHandle");

    nb::class_<SharedEvent, Event>(m, "SharedEvent")
        .def("signal", &SharedEvent::signal, nb::arg("value"))
        .def_prop_ro("signaled_value", &SharedEvent::signaled_value)
        .def("wait", &SharedEvent::wait,
             nb::arg("value"), nb::arg("timeout_ms"),
             // Blocks the calling thread on Metal's own condition variable
             // (waitUntilSignaledValue) -- same rationale as CommandBuffer::
             // commit(wait=True) for releasing the GIL around a blocking call.
             nb::call_guard<nb::gil_scoped_release>())
        .def("new_shared_event_handle", &SharedEvent::new_shared_event_handle,
             nb::rv_policy::take_ownership);

    nb::class_<CommandBuffer>(m, "CommandBuffer")
        .def("commit", &CommandBuffer::commit,
             nb::arg("wait") = true,
             // Same rationale as Pipeline::run's GIL release below -- this
             // blocks on waitUntilCompleted() when wait=True.
             nb::call_guard<nb::gil_scoped_release>())
        .def("encode_wait_for_event", &CommandBuffer::encode_wait_for_event,
             nb::arg("event"), nb::arg("value"),
             // Unlike buffers/textures/samplers (which Metal's own encoder
             // retains internally once bound -- see Pipeline::run's binding
             // below), MTLEvent isn't an MTLResource and isn't documented as
             // being retained by encodeWaitForEvent/encodeSignalEvent -- keep
             // this CommandBuffer's Python wrapper holding a reference to
             // `event` for as long as the CommandBuffer itself is alive, so
             // a caller passing a throwaway `device.event()` inline can't
             // have it garbage-collected (and the underlying MTL::Event
             // released) while a wait=False commit is still pending on it.
             nb::keep_alive<1, 2>())
        .def("encode_signal_event", &CommandBuffer::encode_signal_event,
             nb::arg("event"), nb::arg("value"),
             nb::keep_alive<1, 2>());  // same rationale as encode_wait_for_event above

    nb::class_<Pipeline>(m, "Pipeline")
        .def("run", &Pipeline::run,
             nb::arg("buffers"), nb::arg("textures"), nb::arg("samplers"),
             nb::arg("grid"), nb::arg("wait") = true,
             nb::arg("command_buffer") = nullptr,
             nb::arg("threadgroup") = std::nullopt,
             nb::arg("wait_fences") = std::vector<Fence*>{},
             nb::arg("signal_fences") = std::vector<Fence*>{},
             // Pipeline::run touches only raw C++/Metal state after argument
             // conversion (no PyObject* access), so it's safe to release the
             // GIL for the whole call -- otherwise a wait=True dispatch fully
             // blocks every other Python thread for the entire GPU round
             // trip (confirmed: a background thread made ~zero progress
             // during the call, not just some).
             nb::call_guard<nb::gil_scoped_release>(),
             // Same rationale as CommandBuffer::encode_wait_for_event's
             // keep_alive above: MTL::Fence isn't an MTLResource, so there's
             // no guaranteed internal retain from waitForFence/updateFence
             // to lean on the way setBuffer/setTexture's does. Ties each
             // list's lifetime (and thus every Fence element it holds a
             // reference to) to this Pipeline, which callers already keep
             // alive for as long as they keep dispatching -- comfortably
             // longer than any one in-flight dispatch needs.
             nb::keep_alive<1, 9>(), nb::keep_alive<1, 10>())
        .def("thread_execution_width",      &Pipeline::thread_execution_width)
        .def("max_threads_per_threadgroup", &Pipeline::max_threads_per_threadgroup);
}
