//! ARC two-try evaluation with augmentation voting.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::augment::AugmentPipeline;
use crate::data::{detokenize_grid, tokenize_grid, ArcTask};
use crate::recursion::TrmModel;

#[derive(Debug, Clone, Copy)]
pub struct EvalResult {
    pub total: usize,
    pub exact: usize,
}

impl EvalResult {
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.exact as f64 / self.total as f64
        }
    }
}

pub fn evaluate_arc_tasks(
    model: &TrmModel,
    device: &Device,
    tasks: &[(String, ArcTask)],
    n_augmentations: usize,
    two_try: bool,
) -> Result<EvalResult> {
    let mut total = 0usize;
    let mut exact = 0usize;
    let attempts = if two_try { 2 } else { 1 };

    for (_name, task) in tasks {
        for (test_idx, x_grid) in task.test_inputs.iter().enumerate() {
            let gt = &task.test_outputs[test_idx];
            let mut hit = false;
            for attempt in 0..attempts {
                let seed = 42 + attempt as u64 * 0x9E3779B97F4A7C15;
                let pipeline = AugmentPipeline {
                    n_augmentations,
                    seed,
                    parallel: true,
                };
                let augs = pipeline.augment_grid(x_grid);
                let mut preds = Vec::with_capacity(augs.len());
                for (x_aug, t) in augs {
                    let rows = x_aug.len();
                    let cols = if rows > 0 { x_aug[0].len() } else { 0 };
                    let x_tok = tokenize_grid(&x_aug);
                    let x_tok = Tensor::from_vec(x_tok, (1, rows * cols), device)?
                        .to_dtype(DType::U32)?;
                    let puzzle_id = Tensor::from_vec(vec![task.puzzle_id], 1, device)?
                        .to_dtype(DType::U32)?;
                    let pred_tok =
                        model.forward_inference(&x_tok, rows * cols, Some(&puzzle_id), None)?;
                    let pred_tok = pred_tok.flatten_all()?.to_vec1::<u32>()?;
                    let pred_grid = detokenize_grid(&pred_tok, rows, cols);
                    preds.push((pred_grid, t));
                }
                let voted = pipeline.majority_vote(&preds);
                if &voted == gt {
                    hit = true;
                    break;
                }
            }
            if hit {
                exact += 1;
            }
            total += 1;
        }
    }
    Ok(EvalResult { total, exact })
}
