//! CPU clones of the CUDA kernels in `src/kernels/*.cu`.
//!
//! These exist so the inference GEMM path is one algorithm on CPU and GPU.
//! Numerics follow the CUDA loops (integer `in_dim / 4` packing, no K remainder).

use crate::quantize::TernaryPacked;

fn unpack_w(q: u8) -> f32 {
    match q & 0x03 {
        0 => -1.0,
        2 => 1.0,
        _ => 0.0,
    }
}

/// `ternary_matmul` / `ternary_matmul_stack` inner loop.
pub fn ternary_matmul_stack(input: &[f32], pack: &TernaryPacked, tokens: usize) -> Vec<f32> {
    assert_eq!(input.len(), tokens * pack.in_dim);
    let mut output = vec![0f32; tokens * pack.out_dim];
    let in_dim = pack.in_dim;
    let out_dim = pack.out_dim;
    let pk_max = in_dim / 4;
    for token_idx in 0..tokens {
        let row_in = &input[token_idx * in_dim..token_idx * in_dim + in_dim];
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            let row_w = &pack.packed[o * pk_max..(o + 1) * pk_max];
            for pk in 0..pk_max {
                let byte = row_w[pk];
                for j in 0..4 {
                    let q = (byte >> (j * 2)) & 0x03;
                    acc += unpack_w(q) * row_in[pk * 4 + j];
                }
            }
            output[token_idx * out_dim + o] = acc * pack.alphas[o];
        }
    }
    output
}

pub fn ternary_matmul(
    input: &[f32],
    pack: &TernaryPacked,
    batch: usize,
    apply_relu: bool,
) -> Vec<f32> {
    let mut out = ternary_matmul_stack(input, pack, batch);
    if apply_relu {
        for v in &mut out {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }
    out
}

/// `swiglu_fused`: silu(gate) * up
pub fn swiglu_fused(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

/// Row-wise layer norm matching `activations.cu` (mean/var over cols, population).
pub fn layer_norm(input: &[f32], gamma: Option<&[f32]>, beta: Option<&[f32]>, rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), rows * cols);
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|x| {
            let d = x - mean;
            d * d
        }).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            let g = gamma.map(|g| g[c]).unwrap_or(1.0);
            let b = beta.map(|b| b[c]).unwrap_or(0.0);
            out[r * cols + c] = (row[c] - mean) * inv_std * g + b;
        }
    }
    out
}

/// `dense_matmul_bias`: C[M,N] = A[M,K] @ B[K,N] + bias[N]
pub fn dense_matmul_bias(a: &[f32], b: &[f32], bias: Option<&[f32]>, m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = bias.map(|bias| bias[col]).unwrap_or(0.0);
            for kk in 0..k {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

pub fn bias_add(data: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(bias.len(), cols);
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] += bias[c];
        }
    }
}

fn linear_with_bias(
    input: &[f32],
    pack: &TernaryPacked,
    bias: Option<&[f32]>,
    tokens: usize,
) -> Vec<f32> {
    let mut out = ternary_matmul_stack(input, pack, tokens);
    if let Some(b) = bias {
        bias_add(&mut out, b, tokens, pack.out_dim);
    }
    out
}

/// gate / value / silu*up / down — same graph as the CUDA fused FFN.
pub fn packed_swiglu_ffn(
    xs: &[f32],
    tokens: usize,
    gate: &TernaryPacked,
    gate_bias: Option<&[f32]>,
    value: &TernaryPacked,
    value_bias: Option<&[f32]>,
    down: &TernaryPacked,
    down_bias: Option<&[f32]>,
) -> Vec<f32> {
    let g = linear_with_bias(xs, gate, gate_bias, tokens);
    let v = linear_with_bias(xs, value, value_bias, tokens);
    let hidden = swiglu_fused(&g, &v);
    linear_with_bias(&hidden, down, down_bias, tokens)
}

pub fn transpose_bshd_to_bhsd(src: &[f32], b: usize, s: usize, h: usize, d: usize) -> Vec<f32> {
    let mut dst = vec![0f32; b * h * s * d];
    for bi in 0..b {
        for si in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    let src_idx = ((bi * s + si) * h + hi) * d + di;
                    let dst_idx = ((bi * h + hi) * s + si) * d + di;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    dst
}

pub fn transpose_bhsd_to_bsd(src: &[f32], b: usize, h: usize, s: usize, d: usize) -> Vec<f32> {
    let mut dst = vec![0f32; b * s * h * d];
    for bi in 0..b {
        for hi in 0..h {
            for si in 0..s {
                for di in 0..d {
                    let src_idx = ((bi * h + hi) * s + si) * d + di;
                    let dst_idx = (bi * s + si) * (h * d) + hi * d + di;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    dst
}

/// In-place RoPE on `[B, H, S, Dh]` matching `attention.cu`.
pub fn apply_rope(q_or_k: &mut [f32], cos: &[f32], sin: &[f32], batch: usize, heads: usize, seq: usize, head_dim: usize) {
    let half = head_dim / 2;
    let total = batch * heads * seq;
    for token_idx in 0..total {
        let s = token_idx % seq;
        let offset = token_idx * head_dim;
        for i in 0..half {
            let x1 = q_or_k[offset + i];
            let x2 = q_or_k[offset + half + i];
            let c = cos[s * half + i];
            let sn = sin[s * half + i];
            q_or_k[offset + i] = x1 * c - x2 * sn;
            q_or_k[offset + half + i] = x1 * sn + x2 * c;
        }
    }
}

/// Softmax attention matching `fused_attention` in `attention.cu`
/// (CPU two-pass; CUDA uses an algebraically equivalent online softmax).
pub fn fused_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    b: usize,
    h: usize,
    h_kv: usize,
    s: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; b * h * s * d];
    let repeat = if h_kv == h { 1 } else { h / h_kv.max(1) };
    for bi in 0..b {
        for hi in 0..h {
            let kv_h = if h_kv == h { hi } else { hi / repeat.max(1) };
            for q_pos in 0..s {
                let q_off = ((bi * h + hi) * s + q_pos) * d;
                let mut max_score = f32::NEG_INFINITY;
                for k_pos in 0..s {
                    let k_off = ((bi * h_kv + kv_h) * s + k_pos) * d;
                    let mut score = 0.0f32;
                    for i in 0..d {
                        score += q[q_off + i] * k[k_off + i];
                    }
                    score *= scale;
                    if let Some(m) = mask {
                        score += m[(bi * s + q_pos) * s + k_pos];
                    }
                    if score > max_score {
                        max_score = score;
                    }
                }
                let mut sum_w = 0.0f32;
                let mut acc = vec![0f32; d];
                for k_pos in 0..s {
                    let k_off = ((bi * h_kv + kv_h) * s + k_pos) * d;
                    let v_off = k_off;
                    let mut score = 0.0f32;
                    for i in 0..d {
                        score += q[q_off + i] * k[k_off + i];
                    }
                    score *= scale;
                    if let Some(m) = mask {
                        score += m[(bi * s + q_pos) * s + k_pos];
                    }
                    let w = (score - max_score).exp();
                    sum_w += w;
                    for i in 0..d {
                        acc[i] += w * v[v_off + i];
                    }
                }
                let denom = sum_w + 1e-12;
                let o_off = q_off;
                for i in 0..d {
                    out[o_off + i] = acc[i] / denom;
                }
            }
        }
    }
    out
}

pub fn packed_mha(
    xs: &[f32],
    b: usize,
    s: usize,
    q: &TernaryPacked,
    q_bias: Option<&[f32]>,
    k: &TernaryPacked,
    k_bias: Option<&[f32]>,
    v: &TernaryPacked,
    v_bias: Option<&[f32]>,
    o: &TernaryPacked,
    o_bias: Option<&[f32]>,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let tokens = b * s;
    let qv = linear_with_bias(xs, q, q_bias, tokens);
    let kv = linear_with_bias(xs, k, k_bias, tokens);
    let vv = linear_with_bias(xs, v, v_bias, tokens);
    let mut qh = transpose_bshd_to_bhsd(&qv, b, s, n_heads, head_dim);
    let mut kh = transpose_bshd_to_bhsd(&kv, b, s, n_kv, head_dim);
    apply_rope(&mut qh, rope_cos, rope_sin, b, n_heads, s, head_dim);
    apply_rope(&mut kh, rope_cos, rope_sin, b, n_kv, s, head_dim);
    let vh = transpose_bshd_to_bhsd(&vv, b, s, n_kv, head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let ctx = fused_attention(&qh, &kh, &vh, mask, b, n_heads, n_kv, s, head_dim, scale);
    let ctx_bsd = transpose_bhsd_to_bsd(&ctx, b, n_heads, s, head_dim);
    linear_with_bias(&ctx_bsd, o, o_bias, tokens)
}
