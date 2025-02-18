#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace cg = cooperative_groups;
using namespace nvcuda;

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    }}

template <typename T, int N>
struct AlignedType {
    __align__(N * sizeof(T)) T data[N];
};

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
    return __nv_cvt_fp8_to_fp32(val);
}

__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8(float val) {
    return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
}
