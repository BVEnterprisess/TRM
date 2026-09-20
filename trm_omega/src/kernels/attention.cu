// CUDA Kernels for TRM-Omega: Attention and RoPE
#include <cuda_runtime.h>
#include <math.h>

extern "C" {

__global__ void precompute_rope(
    float* cos_out,
    float* sin_out,
    int max_seq,
    int head_dim,
    float base
) {
    int pos = blockIdx.x;
    int i = threadIdx.x;
    int half = head_dim / 2;

    if (pos < max_seq && i < half) {
        float exponent = 2.0f * (float)i / (float)head_dim;
        float theta = (float)pos / powf(base, exponent);
        cos_out[pos * half + i] = cosf(theta);
        sin_out[pos * half + i] = sinf(theta);
    }
}

__global__ void apply_rope(
    float* q_or_k,            // [batch, num_heads, seq_len, head_dim]
    const float* cos_cache,   // [seq_len, head_dim / 2]
    const float* sin_cache,   // [seq_len, head_dim / 2]
    int batch,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int half = head_dim / 2;
    int total = batch * num_heads * seq_len;
    int token_idx = blockIdx.x;
    int i = threadIdx.x;

    if (token_idx >= total || i >= half) return;

    int s = token_idx % seq_len;
    int offset = token_idx * head_dim;

    float x1 = q_or_k[offset + i];
    float x2 = q_or_k[offset + half + i];

    float c = cos_cache[s * half + i];
    float sn = sin_cache[s * half + i];

    q_or_k[offset + i]        = x1 * c - x2 * sn;
    q_or_k[offset + half + i] = x1 * sn + x2 * c;
}

__global__ void transpose_bshd_to_bhsd(
    const float* src,
    float* dst,
    int b,
    int s,
    int h,
    int d
) {
    int total = b * s * h * d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int rem = idx;
    int d_idx = rem % d; rem /= d;
    int h_idx = rem % h; rem /= h;
    int s_idx = rem % s;
    int b_idx = rem / s;

    int dst_idx = ((b_idx * h + h_idx) * s + s_idx) * d + d_idx;
    dst[dst_idx] = src[idx];
}

__global__ void transpose_bhsd_to_bsd(
    const float* src,
    float* dst,
    int b,
    int h,
    int s,
    int d
) {
    int total = b * h * s * d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int rem = idx;
    int d_idx = rem % d; rem /= d;
    int s_idx = rem % s; rem /= s;
    int h_idx = rem % h;
    int b_idx = rem / h;

    int dst_idx = (b_idx * s + s_idx) * (h * d) + (h_idx * d + d_idx);
    dst[dst_idx] = src[idx];
}

// nsys 2026-09-20 (n_L=2 n_sup=2): naive fused_attention was 68.4% of GPU
// kernel time — every thread recomputed the full QK dot twice. This version
// cooperates across D, tiles K/V in shared memory, and uses online softmax.
#ifndef TRM_ATTN_TILE
#define TRM_ATTN_TILE 32
#endif

__device__ __forceinline__ float attn_reduce_sum(float val, float* red, int n) {
    int t = threadIdx.x;
    red[t] = (t < n) ? val : 0.0f;
    __syncthreads();
    for (int stride = 1; stride < n; stride <<= 1) {
        if ((t & ((stride << 1) - 1)) == 0 && (t + stride) < n) {
            red[t] += red[t + stride];
        }
        __syncthreads();
    }
    return red[0];
}

__global__ void fused_attention(
    const float* Q,           // [B, H, S, D]
    const float* K,           // [B, H_kv, S, D]
    const float* V,           // [B, H_kv, S, D]
    const float* mask,        // [B, 1, S, S] or NULL
    float* out,               // [B, H, S, D]
    int B,
    int H,
    int H_kv,
    int S,
    int D,
    float scale
) {
    int bh_s = blockIdx.x;
    int d_idx = threadIdx.x;

    int total_queries = B * H * S;
    if (bh_s >= total_queries) return;

    int q_pos = bh_s % S;
    int rem = bh_s / S;
    int h = rem % H;
    int b = rem / H;

    int kv_h = (H_kv == H) ? h : (h / (H / H_kv));

    const float* q_vec = Q + ((b * H + h) * S + q_pos) * D;
    const float* k_mat = K + (b * H_kv + kv_h) * S * D;
    const float* v_mat = V + (b * H_kv + kv_h) * S * D;

    extern __shared__ float smem[];
    float* k_s = smem;
    float* v_s = k_s + TRM_ATTN_TILE * D;
    float* red = v_s + TRM_ATTN_TILE * D;

    float q_t = (d_idx < D) ? q_vec[d_idx] : 0.0f;
    float acc = 0.0f;
    float m = -1e20f;
    float l = 0.0f;

    for (int tile = 0; tile < S; tile += TRM_ATTN_TILE) {
        int n = S - tile;
        if (n > TRM_ATTN_TILE) n = TRM_ATTN_TILE;
        for (int row = 0; row < n; ++row) {
            if (d_idx < D) {
                k_s[row * D + d_idx] = k_mat[(tile + row) * D + d_idx];
                v_s[row * D + d_idx] = v_mat[(tile + row) * D + d_idx];
            }
        }
        __syncthreads();

        for (int row = 0; row < n; ++row) {
            float partial = (d_idx < D) ? q_t * k_s[row * D + d_idx] : 0.0f;
            float score = attn_reduce_sum(partial, red, D) * scale;
            if (mask) {
                score += mask[b * S * S + q_pos * S + (tile + row)];
            }
            float m_new = fmaxf(m, score);
            float alpha = expf(m - m_new);
            float w = expf(score - m_new);
            float v_t = (d_idx < D) ? v_s[row * D + d_idx] : 0.0f;
            acc = acc * alpha + w * v_t;
            l = l * alpha + w;
            m = m_new;
        }
        __syncthreads();
    }

    if (d_idx < D) {
        out[((b * H + h) * S + q_pos) * D + d_idx] = acc / (l + 1e-12f);
    }
}

__global__ void dense_matmul_bias(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = bias ? bias[col] : 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

}
