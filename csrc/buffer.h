#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>

namespace mtlpy {

class Buffer {
public:
    Buffer(MTL::Device* device, size_t size_bytes);
    ~Buffer();

    void*        contents_ptr() const;
    size_t       size_bytes()   const { return size_bytes_; }
    MTL::Buffer* mtl()          const { return buf_; }

private:
    MTL::Buffer* buf_;
    size_t       size_bytes_;
};

} // namespace mtlpy
