cmake_minimum_required(VERSION 3.15)
project(mtlpy LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(METAL_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)

pybind11_add_module(_mtlpy
    csrc/metal_impl.mm
    csrc/device.cpp
    csrc/pipeline_cache.cpp
    csrc/buffer.cpp
    csrc/pipeline.cpp
    csrc/bindings.cpp
)

target_include_directories(_mtlpy PRIVATE
    ${METAL_CPP_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_link_libraries(_mtlpy PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
)

install(TARGETS _mtlpy LIBRARY DESTINATION mtlpy)
