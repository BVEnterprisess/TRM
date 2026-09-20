// CUDA Kernels for TRM-Omega: 2-bit ternary matmul & tensor operations
// Architecture: sm_75+ (Turing, Ampere, Ada, Hopper)
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

/**
 * ternary_matmul
 * Unpacks 2-bit weights (-1, 0, +1) and computes GEMM:
 * output = (weights * input) * alphas [+ optional ReLU]
 *
 * Weight format:
 * 2 bits per weight:
 * 00 = -1.0
 * 01 =  0.0
 * 10 = +1.0
 * 11 =  0.0 (unused / pad)
 * 4 weights packed per byte.
 */
__global__ void ternary_matmul(
    const float* __restrict__ input,       // [batch, in_dim]
    const uint8_t* __restrict__ weights,   // [out_dim, in_dim / 4]
    const float* __restrict__ alphas,      // [out_dim]
    float* __restrict__ output,            // [batch, out_dim]
    int in_dim,
    int out_dim,
    int apply_relu
) {
    int b = blockIdx.x;
    int o = threadIdx.x + blockIdx.y * blockDim.x;

    if (o >= out_dim) return;

    float acc = 0.0f;
    const uint8_t* row_w = weights + o * (in_dim / 4);
    const float* row_in = input + b * in_dim;

    int pk_max = in_dim / 4;
    for (int pk = 0; pk < pk_max; ++pk) {
        uint8_t byte = row_w[pk];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t q = (byte >> (j * 2)) & 0x03;
            float w = 0.0f;
            if (q == 0) w = -1.0f;
            else if (q == 2) w = 1.0f;
            acc += w * row_in[pk * 4 + j];
        }
    }

    float out_val = acc * alphas[o];
    if (apply_relu && out_val < 0.0f) {
        out_val = 0.0f;
    }
    output[b * out_dim + o] = out_val;
}

/**
 * ternary_matmul_stack
 * Batched GEMM over 3D sequence tensors [batch, seq, in_dim]
 */
__global__ void ternary_matmul_stack(
    const float* __restrict__ input,       // [total_tokens, in_dim]
    const uint8_t* __restrict__ weights,   // [out_dim, in_dim / 4]
    const float* __restrict__ alphas,      // [out_dim]
    float* __restrict__ output,            // [total_tokens, out_dim]
    int total_tokens,
    int in_dim,
    int out_dim
) {
    int token_idx = blockIdx.x;
    int o = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_idx >= total_tokens || o >= out_dim) return;

    float acc = 0.0f;
    const uint8_t* row_w = weights + o * (in_dim / 4);
    const float* row_in = input + token_idx * in_dim;

    int pk_max = in_dim / 4;
    for (int pk = 0; pk < pk_max; ++pk) {
        uint8_t byte = row_w[pk];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t q = (byte >> (j * 2)) & 0x03;
            float w = (q == 0) ? -1.0f : ((q == 2) ? 1.0f : 0.0f);
            acc += w * row_in[pk * 4 + j];
        }
    }

    output[token_idx * out_dim + o] = acc * alphas[o];
}

}
