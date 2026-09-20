//! # Optimizers
use std::collections::HashMap;
use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor};
use candle_nn::VarMap;

#[derive(Debug, Clone)]
pub struct ParamGroup {
    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub use_atan2: bool,
    pub name_patterns: Vec<String>,
}

impl Default for ParamGroup {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            weight_decay: 1.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            use_atan2: false,
            name_patterns: vec!["*".to_string()],
        }
    }
}

struct AdamState {
    m: Tensor,
    v: Tensor,
    step: u64,
}

pub struct MultiGroupOptimizer {
    groups: Vec<ParamGroup>,
    states: HashMap<String, AdamState>,
    pending: HashMap<String, Tensor>,
    pub current_clip_scale: f64,
    pub loss_scale: f64,
}

impl MultiGroupOptimizer {
    pub fn new(groups: Vec<ParamGroup>) -> Self {
        Self {
            groups,
            states: HashMap::new(),
            pending: HashMap::new(),
            current_clip_scale: 1.0,
            loss_scale: 1.0,
        }
    }

    pub fn trm_default(base_lr: f64, puzzle_lr: f64, weight_decay: f64, puzzle_wd: f64) -> Self {
        let network_group = ParamGroup {
            lr: base_lr,
            weight_decay,
            use_atan2: true,
            name_patterns: vec![
                "shared_network.*".into(),
                "embeddings.token.*".into(),
                "embeddings.segment.*".into(),
                "embeddings.init_*".into(),
                "halt_head.*".into(),
                "trm_block.*".into(),
                "decoder.*".into(),
            ],
            ..Default::default()
        };
        let puzzle_group = ParamGroup {
            lr: puzzle_lr,
            weight_decay: puzzle_wd,
            use_atan2: false,
            name_patterns: vec!["embeddings.puzzle.*".into()],
            ..Default::default()
        };
        Self::new(vec![network_group, puzzle_group])
    }

    fn find_group(&self, name: &str) -> &ParamGroup {
        for g in &self.groups {
            for pat in &g.name_patterns {
                if pattern_matches(pat, name) {
                    return g;
                }
            }
        }
        &self.groups[0]
    }

    /// Fold a backward pass into the pending gradient buffer (keyed by VarMap name).
    pub fn accumulate(&mut self, varmap: &VarMap, grads: &GradStore) -> Result<()> {
        let mut pairs: Vec<(String, Tensor)> = Vec::new();
        {
            let data = varmap.data().lock().unwrap();
            for (name, var) in data.iter() {
                if let Some(g) = grads.get(var.as_tensor()) {
                    pairs.push((name.clone(), g.clone()));
                }
            }
        }
        for (name, g) in pairs {
            if let Some(old) = self.pending.remove(&name) {
                self.pending.insert(name, (old + g)?);
            } else {
                self.pending.insert(name, g);
            }
        }
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Global L2 clip scale over pending grads. Returns `min(1, max_norm / ||g||)`.
    pub fn clip_scale(&mut self, max_norm: f64) -> Result<f64> {
        if max_norm <= 0.0 || self.pending.is_empty() {
            self.current_clip_scale = 1.0;
            return Ok(1.0);
        }
        let mut total_sq = 0.0f64;
        for g in self.pending.values() {
            let sq = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_sq += sq;
        }
        let norm = total_sq.max(0.0).sqrt();
        let scale = if norm > max_norm { max_norm / norm } else { 1.0 };
        self.current_clip_scale = scale;
        Ok(scale)
    }

    pub fn step(&mut self, varmap: &VarMap, clip_scale: f64, loss_scale: f64) -> Result<()> {
        self.loss_scale = loss_scale;
        let inv_loss = if loss_scale.abs() > 1e-12 {
            1.0 / loss_scale
        } else {
            1.0
        };
        let clip = if clip_scale.is_finite() && clip_scale > 0.0 {
            clip_scale
        } else {
            1.0
        };

        let names: Vec<String> = {
            let data = varmap.data().lock().unwrap();
            data.keys().cloned().collect()
        };

        for name in names {
            let g = match self.pending.get(&name) {
                Some(g) => g.clone(),
                None => continue,
            };
            let g = g.affine(inv_loss * clip, 0.0)?;
            let group = self.find_group(&name).clone();

            let (dtype, device, dims, rank) = {
                let data = varmap.data().lock().unwrap();
                let Some(var) = data.get(&name) else {
                    continue;
                };
                (
                    var.dtype(),
                    var.device().clone(),
                    var.dims().to_vec(),
                    var.as_tensor().rank(),
                )
            };

            let state = self.states.entry(name.clone()).or_insert_with(|| AdamState {
                m: Tensor::zeros(dims.as_slice(), dtype, &device).expect("adam m zeros"),
                v: Tensor::zeros(dims.as_slice(), dtype, &device).expect("adam v zeros"),
                step: 0,
            });
            state.step += 1;
            let t = state.step as i32;

            let m = state
                .m
                .affine(group.beta1, 0.0)?
                .add(&g.affine(1.0 - group.beta1, 0.0)?)?;
            let v = state
                .v
                .affine(group.beta2, 0.0)?
                .add(&g.sqr()?.affine(1.0 - group.beta2, 0.0)?)?;
            state.m = m.clone();
            state.v = v.clone();

            let bc1 = 1.0 - group.beta1.powi(t);
            let bc2 = 1.0 - group.beta2.powi(t);
            let m_hat = m.affine(1.0 / bc1.max(1e-16), 0.0)?;
            let v_hat = v.affine(1.0 / bc2.max(1e-16), 0.0)?;
            let v_sqrt = v_hat.sqrt()?;

            let update = if group.use_atan2 {
                tensor_atan2(&m_hat, &v_sqrt)?
            } else {
                m_hat.broadcast_div(&v_sqrt.affine(1.0, group.eps)?)?
            };

            let wd = if rank > 1 { group.weight_decay } else { 0.0 };
            let data = varmap.data().lock().unwrap();
            let var = match data.get(&name) {
                Some(v) => v,
                None => continue,
            };
            let current = var.as_tensor().detach();
            // affine(1,0) can share storage; always allocate a fresh buffer for Var::set.
            let decayed = if wd.abs() > 0.0 {
                current.affine(1.0 - group.lr * wd, 0.0)?
            } else {
                current.copy()?
            };
            let step_t = update.affine(group.lr, 0.0)?;
            let new_p = decayed.sub(&step_t)?;
            var.set(&new_p)?;
        }
        Ok(())
    }

    pub fn set_lrs(&mut self, lrs: &[f64]) {
        for (g, &lr) in self.groups.iter_mut().zip(lrs) {
            g.lr = lr;
        }
    }

    pub fn zero_grad(&mut self, _varmap: &VarMap) -> Result<()> {
        self.pending.clear();
        Ok(())
    }
}

fn tensor_atan2(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    let shape = y.shape().clone();
    let device = y.device();
    let yv = y.flatten_all()?.to_vec1::<f32>()?;
    let xv = x.flatten_all()?.to_vec1::<f32>()?;
    let out: Vec<f32> = yv
        .iter()
        .zip(xv.iter())
        .map(|(a, b)| a.atan2(*b))
        .collect();
    Tensor::from_vec(out, shape, &device)
}

fn pattern_matches(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    pattern == name
}

pub struct CosineScheduler {
    pub base_lrs: Vec<f64>,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub min_lr_ratio: f64,
}

impl CosineScheduler {
    pub fn new(base_lrs: Vec<f64>, warmup_steps: u64, total_steps: u64) -> Self {
        Self {
            base_lrs,
            warmup_steps,
            total_steps,
            min_lr_ratio: 0.01,
        }
    }

    pub fn get_lrs(&self, step: u64) -> Vec<f64> {
        self.base_lrs
            .iter()
            .map(|&base| {
                let min_lr = base * self.min_lr_ratio;
                if step < self.warmup_steps {
                    base * (step as f64 / self.warmup_steps.max(1) as f64)
                } else if step >= self.total_steps {
                    min_lr
                } else {
                    let progress = (step - self.warmup_steps) as f64
                        / (self.total_steps - self.warmup_steps).max(1) as f64;
                    let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                    min_lr + (base - min_lr) * cosine
                }
            })
            .collect()
    }
}

pub fn compute_clip_scale(varmap: &VarMap, max_norm: f64) -> Result<f64> {
    let _ = (varmap, max_norm);
    Ok(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn adamw_step_changes_parameters() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w = vb.get_with_hints(
            (2, 2),
            "shared_network.weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.5,
            },
        )?;
        let x = Tensor::from_vec(vec![0.5f32, -0.25, 0.75, 1.0], (2, 2), &device)?;
        let y = w.matmul(&x)?.sqr()?.sum_all()?;
        let grads = y.backward()?;

        let mut opt = MultiGroupOptimizer::trm_default(1e-2, 1e-2, 0.0, 0.0);
        opt.accumulate(&varmap, &grads)?;
        assert!(opt.pending_count() > 0, "expected grads on shared_network.weight");

        let before = {
            let data = varmap.data().lock().unwrap();
            data.get("shared_network.weight")
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?
        };

        let clip = opt.clip_scale(1.0)?;
        opt.step(&varmap, clip, 1.0)?;
        opt.zero_grad(&varmap)?;

        let after = {
            let data = varmap.data().lock().unwrap();
            data.get("shared_network.weight")
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?
        };

        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "AdamW step must update weights");
        Ok(())
    }

    #[test]
    fn cosine_scheduler_warms_up_then_decays() {
        let sch = CosineScheduler::new(vec![1e-3], 10, 110);
        let lr0 = sch.get_lrs(0)[0];
        let lr_mid_warm = sch.get_lrs(5)[0];
        let lr_end_warm = sch.get_lrs(10)[0];
        let lr_late = sch.get_lrs(110)[0];
        assert!(lr0 < lr_mid_warm);
        assert!((lr_end_warm - 1e-3).abs() < 1e-12);
        assert!(lr_late < lr_end_warm);
    }
}
