//! # Loss Functions
//!
//! - Stable cross-entropy per-position
//! - Deep supervision aggregated loss
//! - Halting penalty for ACT
//! - Exact-match accuracy metric

use candle_core::{DType, Device, Result, Tensor, D};

pub fn cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: Option<u32>,
) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;
    let tgt = targets.to_dtype(DType::U32)?;
    let tgt_i64 = tgt.to_dtype(DType::I64)?;
    let tgt_expanded = tgt_i64.unsqueeze(D::Minus1)?;
    let gathered = log_probs.gather(&tgt_expanded, D::Minus1)?;
    let gathered = gathered.squeeze(D::Minus1)?;
    let nll = gathered.neg()?;

    match ignore_index {
        Some(ign) => {
            let mask = tgt.ne(ign as f64)?.to_dtype(DType::F32)?;
            let masked = nll.broadcast_mul(&mask)?;
            let sum = masked.sum_all()?;
            let count = mask.sum_all()?.clamp(1.0, f64::MAX)?;
            sum.broadcast_div(&count)
        }
        None => nll.mean_all(),
    }
}

pub fn deep_supervision_loss(
    intermediates: &[Tensor],
    targets: &Tensor,
    weights: Option<&[f64]>,
    ignore_index: Option<u32>,
) -> Result<Tensor> {
    let n = intermediates.len();
    if n == 0 {
        return Tensor::new(0f32, targets.device());
    }

    let default_w = 1.0 / n as f64;
    let device = targets.device();
    let mut total = Tensor::new(0f32, device)?;

    for (i, logits) in intermediates.iter().enumerate() {
        let w = weights.map(|ws| ws[i]).unwrap_or(default_w);
        let ce = cross_entropy(logits, targets, ignore_index)?;
        total = (&total + &ce.affine(w, 0.0)?)?;
    }

    Ok(total)
}

pub fn linear_increasing_weights(n: usize) -> Vec<f64> {
    let denom: f64 = (1..=n).map(|k| k as f64).sum();
    (1..=n).map(|k| k as f64 / denom).collect()
}

pub fn last_step_only_weights(n: usize) -> Vec<f64> {
    let mut w = vec![0.0; n];
    if n > 0 {
        w[n - 1] = 1.0;
    }
    w
}

pub fn halting_ponder_loss(halt_probs: &[Tensor], lambda: f64) -> Result<Tensor> {
    if halt_probs.is_empty() {
        return Tensor::new(0f32, &Device::Cpu);
    }
    let device = halt_probs[0].device();
    let mut ponder = Tensor::new(0f32, device)?;

    for p in halt_probs.iter() {
        let remainder = p.neg()?.affine(1.0, 1.0)?;
        let step_cost = remainder.mean_all()?;
        ponder = (&ponder + &step_cost)?;
    }

    ponder.affine(lambda, 0.0)
}

pub fn exact_match_accuracy(preds: &[Vec<u32>], targets: &[Vec<u32>]) -> f64 {
    if preds.is_empty() {
        return 0.0;
    }
    let correct = preds.iter().zip(targets).filter(|(p, t)| p == t).count();
    correct as f64 / preds.len() as f64
}

pub fn token_accuracy(preds: &[Vec<u32>], targets: &[Vec<u32>]) -> f64 {
    let mut correct = 0usize;
    let mut total = 0usize;
    for (p, t) in preds.iter().zip(targets) {
        for (a, b) in p.iter().zip(t) {
            if a == b {
                correct += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn cross_entropy_ignores_pad_and_is_finite() -> Result<()> {
        let dev = Device::Cpu;
        // logits [1, 3, 4]: prefer class 1 at every position
        let mut logits = vec![0f32; 12];
        for s in 0..3 {
            logits[s * 4 + 1] = 5.0;
        }
        let logits = Tensor::from_vec(logits, (1, 3, 4), &dev)?;
        let targets = Tensor::from_vec(vec![1u32, 3, 1], (1, 3), &dev)?;
        let loss = cross_entropy(&logits, &targets, Some(3))?;
        let v = loss.to_scalar::<f32>()?;
        assert!(v.is_finite());
        assert!(v < 0.1, "correct class should be cheap, got {v}");
        Ok(())
    }

    #[test]
    fn deep_sup_weights_sum_and_last_only() {
        let w = linear_increasing_weights(4);
        let s: f64 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
        assert!(w[3] > w[0]);
        let last = last_step_only_weights(3);
        assert_eq!(last, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn exact_and_token_accuracy() {
        let preds = vec![vec![1u32, 2, 3], vec![1, 1, 1]];
        let tgts = vec![vec![1u32, 2, 3], vec![1, 0, 1]];
        assert!((exact_match_accuracy(&preds, &tgts) - 0.5).abs() < 1e-12);
        assert!((token_accuracy(&preds, &tgts) - 5.0 / 6.0).abs() < 1e-12);
        assert_eq!(exact_match_accuracy(&[], &[]), 0.0);
    }
}
