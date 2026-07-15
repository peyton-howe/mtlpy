#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "device.h"
#include "buffer.h"
#include "pipeline.h"
#include "sampler.h"
#include "texture.h"

namespace py = pybind11;
using namespace mtlpy;

PYBIND11_MODULE(_mtlpy, m) {
    m.doc() = "Apple Metal GPU compute bindings";

    m.def("list_devices", &Device::available_device_names);

    py::class_<Device>(m, "Device")
        .def(py::init<int>(), py::arg("index") = -1)
        .def("create_buffer", &Device::create_buffer,
             py::arg("size_bytes"),
             py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>())   // keep Device alive while Buffer is alive
        .def("compile", &Device::compile,
             py::arg("source"), py::arg("function_name"),
             py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>())   // keep Device alive while Pipeline is alive
        .def("max_threads_per_threadgroup", &Device::max_threads_per_threadgroup)
        .def("flush_cache", &Device::flush_cache)
        .def("create_texture", &Device::create_texture,
             py::arg("dims"), py::arg("pixel_format"),
             py::arg("width"), py::arg("height"), py::arg("depth"),
             py::arg("usage"), py::arg("private_storage"),
             py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>())   // keep Device alive while Texture is alive
        .def("blit_upload_texture", &Device::blit_upload_texture,
             py::arg("buf"), py::arg("offset"), py::arg("tex"),
             py::arg("bytes_per_row"), py::arg("bytes_per_image"), py::arg("wait"),
             // Same rationale as Pipeline::run's GIL release above -- this
             // blocks on waitUntilCompleted() when wait=True.
             py::call_guard<py::gil_scoped_release>())
        .def("optimize_texture_for_gpu_access", &Device::optimize_texture_for_gpu_access,
             py::arg("tex"), py::arg("wait"),
             py::call_guard<py::gil_scoped_release>())
        .def("copy_texture", &Device::copy_texture,
             py::arg("src"), py::arg("dst"), py::arg("wait"),
             py::call_guard<py::gil_scoped_release>())
        .def("create_sampler", &Device::create_sampler,
             py::arg("linear"), py::arg("repeat"),
             py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>());  // keep Device alive while Sampler is alive

    py::class_<Buffer>(m, "Buffer")
        .def_property_readonly("data_ptr", [](const Buffer& b) {
            return reinterpret_cast<uintptr_t>(b.contents_ptr());
        })
        .def_property_readonly("size_bytes", &Buffer::size_bytes);

    py::class_<Texture>(m, "Texture")
        .def("upload", [](Texture& t, py::buffer data, size_t bytes_per_row, size_t bytes_per_image) {
            py::buffer_info info = data.request();
            t.upload(info.ptr, bytes_per_row, bytes_per_image);
        }, py::arg("data"), py::arg("bytes_per_row"), py::arg("bytes_per_image"))
        .def("download", [](const Texture& t, size_t nbytes, size_t bytes_per_row, size_t bytes_per_image) {
            // PyBytes_FromStringAndSize(nullptr, n) allocates an n-byte
            // bytes object *without* zero-filling or copying into it --
            // unlike building a std::string (zero-init) and then handing it
            // to py::bytes (a second full copy), this touches the payload
            // exactly once, via getBytes() writing straight into the
            // PyBytesObject's own storage. Wrap it in py::bytes immediately
            // so it's exception-safe if t.download() throws.
            PyObject* obj = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)nbytes);
            if (!obj)
                throw py::error_already_set();
            py::bytes result = py::reinterpret_steal<py::bytes>(obj);
            t.download(PyBytes_AS_STRING(obj), bytes_per_row, bytes_per_image);
            return result;
        }, py::arg("nbytes"), py::arg("bytes_per_row"), py::arg("bytes_per_image"))
        .def_property_readonly("width",  &Texture::width)
        .def_property_readonly("height", &Texture::height)
        .def_property_readonly("depth",  &Texture::depth)
        .def_property_readonly("dims",   &Texture::dims)
        .def_property_readonly("is_private", &Texture::is_private);

    py::class_<Sampler>(m, "Sampler");

    py::class_<Pipeline>(m, "Pipeline")
        .def("run", &Pipeline::run,
             py::arg("buffers"), py::arg("textures"), py::arg("samplers"),
             py::arg("grid"), py::arg("wait") = true,
             // Pipeline::run touches only raw C++/Metal state after argument
             // conversion (no PyObject* access), so it's safe to release the
             // GIL for the whole call -- otherwise a wait=True dispatch fully
             // blocks every other Python thread for the entire GPU round
             // trip (confirmed: a background thread made ~zero progress
             // during the call, not just some).
             py::call_guard<py::gil_scoped_release>())
        .def("thread_execution_width",      &Pipeline::thread_execution_width)
        .def("max_threads_per_threadgroup", &Pipeline::max_threads_per_threadgroup);
}
