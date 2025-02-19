#include "cuda_utils.cuh"

__global__ void compute_volatility(
    const float* returns,
    float* volatility,
    int window_size,
    float decay_factor
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f, sum_sq = 0.0f;

    for(int i = 0; i < window_size; ++i) {
        float ret = returns[tid + i];
        float weight = expf(-decay_factor * i);
        sum += weight * ret;
        sum_sq += weight * ret * ret;
    }

    volatility[tid] = sqrtf((sum_sq - (sum * sum)/window_size) / window_size);
}
