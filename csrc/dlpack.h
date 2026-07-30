#pragma once
#include <cstdint>

// Minimal vendored subset of the DLPack stable ABI
// (https://github.com/dmlc/dlpack) -- just the pieces Buffer::_dlpack_capsule
// (see bindings.cpp) needs to build a DLManagedTensor. Field layout must
// match upstream dlpack.h exactly: this struct is read as raw memory by
// other compiled extensions (e.g. MLX, NumPy, PyTorch) via the "dltensor"
// PyCapsule protocol, not through any shared header.

extern "C" {

typedef enum {
    kDLCPU        = 1,
    kDLCUDA       = 2,
    kDLCUDAHost   = 3,
    kDLOpenCL     = 4,
    kDLVulkan     = 7,
    kDLMetal      = 8,
    kDLVPI        = 9,
    kDLROCM       = 10,
    kDLROCMHost   = 11,
    kDLExtDev     = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI     = 14,
    kDLWebGPU     = 15,
    kDLHexagon    = 16,
} DLDeviceType;

typedef struct {
    int32_t device_type;
    int32_t device_id;
} DLDevice;

// DLDataType.code values (DLDataTypeCode), matching upstream's full set --
// mtlpy only ever produces kDLInt/kDLUInt/kDLFloat/kDLBool in practice (see
// utils.to_dlpack_dtype), kDLBfloat/kDLComplex are here for ABI completeness
// only and are currently unreachable from this codebase.
typedef enum {
    kDLInt     = 0U,
    kDLUInt    = 1U,
    kDLFloat   = 2U,
    kDLBfloat  = 4U,
    kDLComplex = 5U,
    kDLBool    = 6U,
} DLDataTypeCode;

typedef struct {
    uint8_t  code;
    uint8_t  bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void*      data;
    DLDevice   device;
    int32_t    ndim;
    DLDataType dtype;
    int64_t*   shape;
    int64_t*   strides;
    uint64_t   byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void*    manager_ctx;
    void     (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

} // extern "C"
