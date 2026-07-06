#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "device.h"
#include "buffer.h"
#include "pipeline.h"

namespace py = pybind11;
using namespace mtlpy;

PYBIND11_MODULE(_mtlpy, m) {
    m.doc() = "Apple Metal GPU compute bindings";

    py::class_<Device>(m, "Device")
        .def(py::init<>())
        .def("create_buffer", &Device::create_buffer,
             py::arg("size_bytes"),
             py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>())   // keep Device alive while Buffer is alive
        .def("compile", &Device::compile,
             py::arg("source"), py::arg("function_name"),
             py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>())   // keep Device alive while Pipeline is alive
        .def("max_threads_per_threadgroup", &Device::max_threads_per_threadgroup);

    py::class_<Buffer>(m, "Buffer")
        .def_property_readonly("data_ptr", [](const Buffer& b) {
            return reinterpret_cast<uintptr_t>(b.contents_ptr());
        })
        .def_property_readonly("size_bytes", &Buffer::size_bytes);

    py::class_<Pipeline>(m, "Pipeline")
        .def("run", &Pipeline::run,
             py::arg("buffers"), py::arg("grid"), py::arg("wait") = true,
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
