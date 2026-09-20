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
    int bh_s = blockIdx.x; // maps to batch, head, and query position
    int d_idx = threadIdx.x;

    int total_queries = B * H * S;
    if (bh_s >= total_queries || d_idx >= D) return;

    int q_pos = bh_s % S;
    int rem = bh_s / S;
    int h = rem % H;
    int b = rem / H;

    int kv_h = (H_kv == H) ? h : (h / (H / H_kv));

    const float* q_vec = Q + ((b * H + h) * S + q_pos) * D;
    const float* k_mat = K + (b * H_kv + kv_h) * S * D;
    const float* v_mat = V + (b * H_kv + kv_h) * S * D;

    float acc = 0.0f;
    float max_score = -1e20f;

    // Local scores buffer in shared memory per thread/block if needed
    // Single-pass softmax accumulation
    float sum_weights = 0.0f;
    for (int k_pos = 0; k_pos < S; ++k_pos) {
        float score = 0.0f;
        for (int i = 0; i < D; ++i) {
            score += q_vec[i] * k_mat[k_pos * D + i];
        }
        score *= scale;
        if (mask) {
            score += mask[b * S * S + q_pos * S + k_pos];
        }
        if (score > max_score) {
            max_score = score;
        }
    }

    for (int k_pos = 0; k_pos < S; ++k_pos) {
        float score = 0.0f;
        for (int i = 0; i < D; ++i) {
            score += q_vec[i] * k_mat[k_pos * D + i];
        }
        score *= scale;
        if (mask) {
            score += mask[b * S * S + q_pos * S + k_pos];
        }
        float w = expf(score - max_score);
        sum_weights += w;
        acc += w * v_mat[k_pos * D + d_idx];
    }

    out[((b * H + h) * S + q_pos) * D + d_idx] = acc / (sum_weights + 1e-12f);
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
