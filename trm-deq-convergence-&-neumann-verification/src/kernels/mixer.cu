// CUDA Kernels for TRM-Omega: MLP-Mixer blocks & transpositions
#include <cuda_runtime.h>
#include <math.h>

extern "C" {

__global__ void transpose_bsd_to_bds(
    const float* src,
    float* dst,
    int b,
    int s,
    int d
) {
    int total = b * s * d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d_idx = idx % d;
    int rem = idx / d;
    int s_idx = rem % s;
    int b_idx = rem / s;

    int dst_idx = (b_idx * d + d_idx) * s + s_idx;
    dst[dst_idx] = src[idx];
}

__global__ void transpose_bds_to_bsd(
    const float* src,
    float* dst,
    int b,
    int d,
    int s
) {
    int total = b * d * s;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int s_idx = idx % s;
    int rem = idx / s;
    int d_idx = rem % d;
    int b_idx = rem / d;

    int dst_idx = (b_idx * s + s_idx) * d + d_idx;
    dst[dst_idx] = src[idx];
}

__global__ void swiglu_up_fused(
    const float* x,
    const float* w_gate,
    const float* w_up,
    float* out,
    int tokens,
    int in_dim,
    int hidden_dim
) {
    int tok = blockIdx.x;
    int h = threadIdx.x + blockIdx.y * blockDim.x;

    if (tok >= tokens || h >= hidden_dim) return;

    const float* x_vec = x + tok * in_dim;
    float g_val = 0.0f;
    float u_val = 0.0f;

    for (int i = 0; i < in_dim; ++i) {
        g_val += x_vec[i] * w_gate[i * hidden_dim + h];
        u_val += x_vec[i] * w_up[i * hidden_dim + h];
    }

    float silu_g = g_val / (1.0f + expf(-g_val));
    out[tok * hidden_dim + h] = silu_g * u_val;
}

__global__ void dense_down(
    const float* inter,
    const float* w_down,
    float* out,
    int tokens,
    int hidden_dim,
    int out_dim
) {
    int tok = blockIdx.x;
    int o = threadIdx.x + blockIdx.y * blockDim.x;

    if (tok >= tokens || o >= out_dim) return;

    const float* in_vec = inter + tok * hidden_dim;
    float acc = 0.0f;
    for (int h = 0; h < hidden_dim; ++h) {
        acc += in_vec[h] * w_down[h * out_dim + o];
    }
    out[tok * out_dim + o] = acc;
}

}
