#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <vector>
#include "device.h"
#include "buffer.h"
#include "command_buffer.h"
#include "dlpack.h"
#include "pipeline.h"
#include "sampler.h"
#include "texture.h"

namespace nb = nanobind;
using namespace mtlpy;

NB_MODULE(_mtlpy, m) {
    m.doc() = "Apple Metal GPU compute bindings";

    m.def("list_devices", &Device::available_device_names);

    nb::class_<Device>(m, "Device")
        .def(nb::init<int>(), nb::arg("index") = -1)
        .def("create_buffer", &Device::create_buffer,
             nb::arg("size_bytes"), nb::arg("storage_mode"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Buffer is alive
        .def("compile", &Device::compile,
             nb::arg("source"), nb::arg("function_name"),
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>())   // keep Device alive while Pipeline is alive
        .def("max_threads_per_threadgroup", &Device::max_threads_per_threadgroup)
        .def("flush_cache", &Device::flush_cache)
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
             nb::rv_policy::take_ownership,
             nb::keep_alive<0, 1>());  // keep Device alive while CommandBuffer is alive

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

    nb::class_<CommandBuffer>(m, "CommandBuffer")
        .def("commit", &CommandBuffer::commit,
             nb::arg("wait") = true,
             // Same rationale as Pipeline::run's GIL release below -- this
             // blocks on waitUntilCompleted() when wait=True.
             nb::call_guard<nb::gil_scoped_release>());

    nb::class_<Pipeline>(m, "Pipeline")
        .def("run", &Pipeline::run,
             nb::arg("buffers"), nb::arg("textures"), nb::arg("samplers"),
             nb::arg("grid"), nb::arg("wait") = true,
             nb::arg("command_buffer") = nullptr,
             nb::arg("threadgroup") = std::nullopt,
             // Pipeline::run touches only raw C++/Metal state after argument
             // conversion (no PyObject* access), so it's safe to release the
             // GIL for the whole call -- otherwise a wait=True dispatch fully
             // blocks every other Python thread for the entire GPU round
             // trip (confirmed: a background thread made ~zero progress
             // during the call, not just some).
             nb::call_guard<nb::gil_scoped_release>())
        .def("thread_execution_width",      &Pipeline::thread_execution_width)
        .def("max_threads_per_threadgroup", &Pipeline::max_threads_per_threadgroup);
}
