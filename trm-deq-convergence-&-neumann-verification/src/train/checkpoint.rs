//! # Checkpointing
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::train::ema::EMA;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMeta {
    pub epoch: usize,
    pub global_step: u64,
    pub best_val_acc: f64,
    pub best_val_loss: f64,
    pub rng_seed: u64,
    pub ema_decay: f64,
    pub ema_warmup: u64,
    pub ema_step: u64,
    pub current_batch_size: usize,
}

impl Default for TrainingMeta {
    fn default() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            best_val_acc: 0.0,
            best_val_loss: f64::MAX,
            rng_seed: 42,
            ema_decay: 0.999,
            ema_warmup: 1000,
            ema_step: 0,
            current_batch_size: 32,
        }
    }
}

pub fn save_checkpoint(
    dir: &Path,
    tag: &str,
    varmap: &VarMap,
    ema: &EMA,
    meta: &TrainingMeta,
) -> Result<()> {
    fs::create_dir_all(dir)?;

    let model_path = dir.join(format!("{tag}.safetensors"));
    varmap.save(&model_path)
        .map_err(|e| anyhow::anyhow!("save model: {e}"))?;

    let ema_path = dir.join(format!("{tag}.ema.safetensors"));
    save_tensor_map(ema.shadow_tensors(), &ema_path)?;

    let meta_path = dir.join(format!("{tag}.meta.json"));
    let json = serde_json::to_string_pretty(meta)?;
    fs::write(&meta_path, json)?;

    log::info!("Checkpoint saved: {}", dir.join(tag).display());
    Ok(())
}

pub fn load_checkpoint(
    dir: &Path,
    tag: &str,
    varmap: &mut VarMap,
    ema: &mut EMA,
) -> Result<TrainingMeta> {
    let model_path = dir.join(format!("{tag}.safetensors"));
    varmap.load(&model_path)
        .map_err(|e| anyhow::anyhow!("load model: {e}"))?;

    let ema_path = dir.join(format!("{tag}.ema.safetensors"));
    if ema_path.exists() {
        log::info!("Loading EMA from {}", ema_path.display());
        let data = candle_core::safetensors::load(&ema_path, &candle_core::Device::Cpu)?;
        let vars = varmap.all_vars();
        let target_device = if let Some(v) = vars.first() {
            v.device().clone()
        } else {
            candle_core::Device::Cpu
        };
        for (name, tensor) in data {
            let tensor = tensor.to_device(&target_device)?;
            ema.shadows_mut().insert(name, tensor);
        }
    }

    let meta_path = dir.join(format!("{tag}.meta.json"));
    let json = fs::read_to_string(&meta_path)
        .with_context(|| format!("reading {}", meta_path.display()))?;
    let meta: TrainingMeta = serde_json::from_str(&json)?;
    Ok(meta)
}

fn save_tensor_map(map: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    candle_core::safetensors::save(map, path)
        .map_err(|e| anyhow::anyhow!("save EMA: {e}"))?;
    Ok(())
}
