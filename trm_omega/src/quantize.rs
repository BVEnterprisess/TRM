//! # Quantization
//!
//! Deployment-oriented ternary quantization.
use std::collections::BTreeMap;
use anyhow::{Context, Result};
use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct TernaryPacked {
    pub out_dim: usize,
    pub in_dim: usize,
    pub packed: Vec<u8>,
    pub alphas: Vec<f32>,
}

pub fn quantize_rowwise_ternary(weight: &[f32], out_dim: usize, in_dim: usize) -> TernaryPacked {
    assert_eq!(weight.len(), out_dim * in_dim);
    assert_eq!(in_dim % 4, 0, "in_dim must be multiple of 4");

    let mut packed = vec![0u8; out_dim * (in_dim / 4)];
    let mut alphas = vec![0f32; out_dim];

    for o in 0..out_dim {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        let abs_mean = row.iter().map(|v| v.abs()).sum::<f32>() / in_dim as f32;
        alphas[o] = abs_mean.max(1e-8);

        for pk in 0..(in_dim / 4) {
            let mut byte = 0u8;
            for j in 0..4 {
                let v = row[pk * 4 + j] / alphas[o];
                let q = if v <= -0.5 {
                    0u8
                } else if v >= 0.5 {
                    2u8
                } else {
                    1u8
                };
                byte |= (q & 0x3) << (j * 2);
            }
            packed[o * (in_dim / 4) + pk] = byte;
        }
    }

    TernaryPacked { out_dim, in_dim, packed, alphas }
}

pub fn unpack_ternary(p: &TernaryPacked) -> Vec<f32> {
    let mut out = vec![0f32; p.out_dim * p.in_dim];
    for o in 0..p.out_dim {
        for pk in 0..(p.in_dim / 4) {
            let byte = p.packed[o * (p.in_dim / 4) + pk];
            for j in 0..4 {
                let q = (byte >> (j * 2)) & 0x3;
                let w = match q {
                    0 => -1.0,
                    1 => 0.0,
                    2 => 1.0,
                    _ => 0.0,
                };
                out[o * p.in_dim + pk * 4 + j] = w * p.alphas[o];
            }
        }
    }
    out
}

pub fn tensor_to_ternary(t: &Tensor) -> Result<TernaryPacked> {
    let (out_dim, in_dim) = t.dims2()?;
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(quantize_rowwise_ternary(&data, out_dim, in_dim))
}

pub fn relative_l2_error(original: &[f32], approx: &[f32]) -> f32 {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (&a, &b) in original.iter().zip(approx.iter()) {
        let d = a - b;
        num += d * d;
        den += a * a;
    }
    (num / den.max(1e-12)).sqrt()
}

pub fn calibrate_activation_scales(samples: &[Vec<f32>]) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }
    let dim = samples[0].len();
    let mut maxv = vec![0f32; dim];
    for s in samples {
        for i in 0..dim {
            maxv[i] = maxv[i].max(s[i].abs());
        }
    }
    maxv.into_iter()
        .map(|m| if m > 1e-8 { 127.0 / m } else { 0.0 })
        .collect()
}

#[derive(Debug, Clone)]
pub struct QuantizedCheckpoint {
    pub tensors: BTreeMap<String, TernaryPacked>,
}

impl QuantizedCheckpoint {
    pub fn from_named_tensors(named: &BTreeMap<String, Tensor>) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (name, t) in named {
            if t.rank() == 2 {
                let pack = tensor_to_ternary(t)
                    .with_context(|| format!("quantizing {name}"))?;
                tensors.insert(name.clone(), pack);
            }
        }
        Ok(Self { tensors })
    }
}

/// Straight-Through Estimator (STE) for 2-bit ternary weights during QAT training.
/// Forward: w_q = alpha * round_ternary(w / alpha)
/// Backward: identity gradient (grad_w = grad_w_q) passed transparently via detached residual.
pub fn ternary_ste(weight: &Tensor) -> candle_core::Result<Tensor> {
    let (out_dim, in_dim) = weight.dims2()?;
    let abs_w = weight.abs()?;
    let alpha = (abs_w.sum_keepdim(1)? / (in_dim as f64))?.clamp(1e-8, 1e4)?;

    // Scale to unit range
    let normalized = weight.broadcast_div(&alpha)?;

    // Discretize to {-1, 0, +1}
    // Hard ternary thresholding at +/- 0.5
    let shape = normalized.shape();
    let flat = normalized.flatten_all()?.to_vec1::<f32>()?;
    let mut discretized = Vec::with_capacity(flat.len());
    for v in flat {
        let q = if v <= -0.5 {
            -1.0f32
        } else if v >= 0.5 {
            1.0f32
        } else {
            0.0f32
        };
        discretized.push(q);
    }
    let discretized_tensor = Tensor::from_vec(discretized, shape, weight.device())?;
    let quantized = discretized_tensor.broadcast_mul(&alpha)?;

    // STE trick: w + detach(quantized - w)
    // Values forward equal `quantized`, but gradients flow directly to `weight`.
    let diff = (quantized - weight)?;
    let diff_detached = diff.detach();
    (weight + diff_detached)
}
