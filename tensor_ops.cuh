#pragma once
#include "cuda_utils.cuh"

template <int M, int N, int K, typename T>
__device__ void tensor_multiply(
    const T* A, const T* B, float* C,
    bool trans_a = false, bool trans_b = false
) {
    wmma::fragment<wmma::matrix_a, M, N, K, T, 
        trans_a ? wmma::col_major : wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, T, 
        trans_b ? wmma::row_major : wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::load_matrix_sync(a_frag, A, trans_a ? K : M);
    wmma::load_matrix_sync(b_frag, B, trans_b ? N : K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}
