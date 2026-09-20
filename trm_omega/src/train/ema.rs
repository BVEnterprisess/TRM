//! # Exponential Moving Average
use std::collections::HashMap;
use candle_core::{Result, Tensor};
use candle_nn::VarMap;

pub struct EMA {
    pub decay: f64,
    pub warmup_steps: u64,
    pub step: u64,
    shadows: HashMap<String, Tensor>,
}

impl EMA {
    pub fn new(decay: f64, warmup_steps: u64) -> Self {
        Self {
            decay,
            warmup_steps,
            step: 0,
            shadows: HashMap::new(),
        }
    }

    pub fn update(&mut self, varmap: &VarMap) -> Result<()> {
        self.step += 1;
        let data = varmap.data().lock().unwrap();
        if self.step <= self.warmup_steps {
            for (name, var) in data.iter() {
                self.shadows
                    .insert(name.clone(), var.as_tensor().detach());
            }
            return Ok(());
        }

        let d = self.decay;
        let od = 1.0 - d;

        for (name, var) in data.iter() {
            let current = var.as_tensor().detach();
            let shadow = self.shadows.entry(name.clone()).or_insert_with(|| current.clone());
            let s_aff = shadow.affine(d, 0.0)?;
            let c_aff = current.affine(od, 0.0)?;
            let new_shadow = (&s_aff + &c_aff)?;
            *shadow = new_shadow;
        }

        Ok(())
    }

    pub fn swap_in(&self, varmap: &VarMap) -> Result<HashMap<String, Tensor>> {
        let mut originals = HashMap::new();
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            originals.insert(name.clone(), var.as_tensor().detach());
            if let Some(shadow) = self.shadows.get(name) {
                var.set(shadow)?;
            }
        }
        Ok(originals)
    }

    pub fn swap_out(originals: &HashMap<String, Tensor>, varmap: &VarMap) -> Result<()> {
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            if let Some(orig) = originals.get(name) {
                var.set(orig)?;
            }
        }
        Ok(())
    }

    pub fn shadow_tensors(&self) -> &HashMap<String, Tensor> {
        &self.shadows
    }

    pub fn shadows_mut(&mut self) -> &mut HashMap<String, Tensor> {
        &mut self.shadows
    }

    pub fn l2_distance(&self, varmap: &VarMap) -> Result<f64> {
        let mut total_sq = 0.0f64;
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            if let Some(shadow) = self.shadows.get(name) {
                let diff = var.as_tensor().sub(shadow)?;
                let sq = diff.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                total_sq += sq;
            }
        }
        Ok(total_sq.sqrt())
    }
}
