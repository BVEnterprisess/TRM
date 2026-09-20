// CUDA Kernels for TRM-Omega: Recursive loop control, halting, and state maintenance
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {

__global__ void halt_confidence(
    const float* y,
    float* out_p,
    int batch,
    int seq_len,
    int dim
) {
    int b = blockIdx.x;
    if (b >= batch) return;

    // Mean pool over seq_len
    float sum = 0.0f;
    const float* y_batch = y + b * seq_len * dim;
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < dim; ++d) {
            sum += fabsf(y_batch[s * dim + d]);
        }
    }
    float mean_val = sum / (float)(seq_len * dim);
    out_p[b] = 1.0f / (1.0f + expf(-mean_val));
}

__global__ void update_halt_mask(
    const float* halt_probs,
    int* halt_mask,
    float threshold,
    int batch
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        if (halt_probs[b] >= threshold) {
            halt_mask[b] = 1;
        }
    }
}

__global__ void deep_supervision_l2(
    const float* cur_logits,
    const float* target_logits,
    float* loss_acc,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float diff = cur_logits[idx] - target_logits[idx];
        atomicAdd(loss_acc, diff * diff);
    }
}

__global__ void slice_copy(
    const float* src,
    float* dst,
    int num_items,
    int item_size,
    int src_stride,
    int dst_stride
) {
    int item = blockIdx.x;
    int offset = threadIdx.x;
    if (item < num_items && offset < item_size) {
        dst[item * dst_stride + offset] = src[item * src_stride + offset];
    }
}

__global__ void write_subseq(
    const float* src,
    float* dst,
    int batch,
    int start_seq,
    int sub_seq,
    int total_seq,
    int dim
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;

    if (b < batch && s < sub_seq && d < dim) {
        int src_idx = (b * sub_seq + s) * dim + d;
        int dst_idx = (b * total_seq + (start_seq + s)) * dim + d;
        dst[dst_idx] = src[src_idx];
    }
}

__global__ void argmax_per_pos(
    const float* logits,
    int32_t* preds,
    int total_tokens,
    int vocab_size
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= total_tokens) return;

    const float* row = logits + tok * vocab_size;
    float max_val = -1e20f;
    int best_v = 0;
    for (int v = 0; v < vocab_size; ++v) {
        if (row[v] > max_val) {
            max_val = row[v];
            best_v = v;
        }
    }
    preds[tok] = best_v;
}

__global__ void add_segment_embed(
    float* x,
    const float* seg_embed,
    int batch,
    int seq_len,
    int dim
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;

    if (b < batch && s < seq_len && d < dim) {
        int idx = (b * seq_len + s) * dim + d;
        x[idx] += seg_embed[d];
    }
}

__global__ void zero_agent_state(float* state, int total_floats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_floats) {
        state[idx] = 0.0f;
    }
}

__global__ void ema_update(
    float* ema_param,
    const float* model_param,
    float decay,
    int total_params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_params) {
        ema_param[idx] = decay * ema_param[idx] + (1.0f - decay) * model_param[idx];
    }
}

}
