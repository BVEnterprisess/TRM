// CUDA Kernels for TRM-Omega: Fused activations, norm, and loss
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {

// SwiGLU: output = (gate * sigmoid(gate)) * up
__global__ void swiglu_fused(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = silu_g * up[idx];
    }
}

__global__ void relu_inplace(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && data[idx] < 0.0f) {
        data[idx] = 0.0f;
    }
}

__global__ void gelu_inplace(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void silu_inplace(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void ewise_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void ewise_add_inplace(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

__global__ void ewise_mul(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void ewise_add_scaled(float* a, const float* b, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += scale * b[idx];
    }
}

__global__ void bias_add(float* data, const float* bias, int rows, int cols) {
    int r = blockIdx.x;
    int c = threadIdx.x + blockIdx.y * blockDim.x;
    if (r < rows && c < cols) {
        data[r * cols + c] += bias[c];
    }
}

__global__ void copy_buffer(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void fill_const(float* dst, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = val;
    }
}

__device__ void layer_norm_impl(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows,
    int cols,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;

    const float* in_row = input + r * cols;
    float* out_row = output + r * cols;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += in_row[c];
    }
    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    float mean = s_sum[0] / cols;

    float sq_diff = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = in_row[c] - mean;
        sq_diff += d * d;
    }
    s_sum[threadIdx.x] = sq_diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    float var = s_sum[0] / cols;
    float inv_std = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float norm = (in_row[c] - mean) * inv_std;
        float g = gamma ? gamma[c] : 1.0f;
        float b = beta ? beta[c] : 0.0f;
        out_row[c] = norm * g + b;
    }
}

__global__ void layer_norm(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows,
    int cols,
    float eps
) {
    layer_norm_impl(input, gamma, beta, output, rows, cols, eps);
}

__global__ void layer_norm_inplace(
    float* data,
    const float* gamma,
    const float* beta,
    int rows,
    int cols,
    float eps
) {
    layer_norm_impl(data, gamma, beta, data, rows, cols, eps);
}

__global__ void softmax_row(float* data, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    float* row = data + r * cols;
    float max_val = -1e20f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        if (row[c] > max_val) max_val = row[c];
    }
    __shared__ float s_val[256];
    s_val[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_val[threadIdx.x] = fmaxf(s_val[threadIdx.x], s_val[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = s_val[0];

    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float exp_v = expf(row[c] - max_val);
        row[c] = exp_v;
        sum += exp_v;
    }
    s_val[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_val[threadIdx.x] += s_val[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / (s_val[0] + 1e-12f);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        row[c] *= inv_sum;
    }
}

__global__ void cross_entropy_loss(
    const float* __restrict__ logits,  // [n, vocab]
    const int32_t* __restrict__ targets, // [n]
    float* __restrict__ losses,        // [n]
    int n,
    int vocab,
    int ignore_index
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int tgt = targets[i];
    if (tgt == ignore_index || tgt < 0 || tgt >= vocab) {
        losses[i] = 0.0f;
        return;
    }

    const float* logit_row = logits + i * vocab;
    float max_l = -1e20f;
    for (int v = 0; v < vocab; ++v) {
        if (logit_row[v] > max_l) max_l = logit_row[v];
    }
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab; ++v) {
        sum_exp += expf(logit_row[v] - max_l);
    }
    float log_sum_exp = max_l + logf(sum_exp + 1e-12f);
    losses[i] = log_sum_exp - logit_row[tgt];
}

__global__ void check_nan_inf(const float* x, int* flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        if (isnan(v)) flags[0] = 1;
        if (isinf(v)) flags[1] = 1;
    }
}

}
