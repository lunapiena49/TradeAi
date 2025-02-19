#include "cuda_utils.cuh"
#include "tensor_ops.cuh"

__global__ void __launch_bounds__(256, 4) process_tick_data(
    const __nv_fp8_e4m3* input,
    float* output,
    const float* filters,
    int seq_len,
    int num_features
) {
    extern __shared__ __align__(128) float smem[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    for(int i = 0; i < 4; ++i) {
        const int offset = tid + i * blockDim.x;
        if(offset < seq_len)
            smem[offset] = fp8_to_float(input[bid * seq_len + offset]);
    }
    __syncthreads();

    tensor_multiply<16, 16, 16, __nv_fp8_e4m3>(
        smem + tid * 16, filters, output + bid * seq_len + tid * 16
    );
}
