//! # Trainer — BRUTAL OPTIMIZED VERSION with DEQ Support
//!
//! - Async data loading with micro-batching
//! - Attention masking for variable sequence lengths
//! - Deep supervision & Halting loss
//! - Cosine LR scheduler & EMA with L2 monitoring
//! - Implicit Function Theorem (IFT) backward for Deep Equilibrium (DEQ)

use std::path::PathBuf;
use std::time::Instant;
use crossbeam_channel::{bounded, Receiver};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::augment::{AugmentCache, AugmentPipeline};
use crate::data::{
    arc_training_examples, detokenize_grid, load_arc_tasks, load_maze_dir, load_sudoku_txt,
    pad_tokens, tokenize_grid, TokenizedExample,
};
use crate::deq::DeqWrapper;
use crate::network::NetworkConfig;
use crate::recursion::{TrmConfig, TrmModel};
use crate::train::checkpoint::{load_checkpoint, save_checkpoint, TrainingMeta};
use crate::train::ema::EMA;
use crate::train::loss::{
    cross_entropy, deep_supervision_loss, exact_match_accuracy, halting_ponder_loss,
    linear_increasing_weights, token_accuracy,
};
use crate::train::optimizer::{CosineScheduler, MultiGroupOptimizer};

// ─── Train Metrics ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct TrainMetrics {
    pub loss: f64,
    pub grad_norm: f64,
    pub ema_l2_distance: f64,
    pub peak_memory_mb: f64,
    pub lr: f64,
    pub steps_used: usize,
    pub batch_size: usize,
    pub per_step_accuracies: Vec<f64>,
}

// ─── TrainConfig ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
    pub task_type: TaskType,
    pub n_augmentations_train: usize,
    pub n_augmentations_eval: usize,
    pub aug_seed: u64,
    pub batch_size: usize,
    pub micro_batch_size: usize,
    pub max_tokens_per_batch: usize,
    pub base_lr: f64,
    pub puzzle_lr: f64,
    pub weight_decay: f64,
    pub puzzle_wd: f64,
    pub grad_clip: f64,
    pub warmup_steps: u64,
    pub max_epochs: usize,
    pub eval_every_epochs: usize,
    pub save_every_epochs: usize,
    pub log_every_steps: usize,
    pub ema_decay: f64,
    pub ema_warmup: u64,
    pub ds_weight_scheme: DsWeightScheme,
    pub halt_lambda: f64,
    pub checkpoint_dir: PathBuf,
    pub resume_tag: Option<String>,
    pub use_fp16: bool,
    pub seed: u64,
    pub no_augment: bool,
    pub num_data_workers: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            task_type: TaskType::Arc,
            n_augmentations_train: 8,
            n_augmentations_eval: 100,
            aug_seed: 42,
            batch_size: 32,
            micro_batch_size: 4,
            max_tokens_per_batch: 32 * 900,
            base_lr: 1e-3,
            puzzle_lr: 1e-4,
            weight_decay: 1.0,
            puzzle_wd: 1.0,
            grad_clip: 1.0,
            warmup_steps: 1000,
            max_epochs: 50_000,
            eval_every_epochs: 5000,
            save_every_epochs: 10_000,
            log_every_steps: 100,
            ema_decay: 0.999,
            ema_warmup: 1000,
            ds_weight_scheme: DsWeightScheme::LinearIncreasing,
            halt_lambda: 0.01,
            checkpoint_dir: PathBuf::from("checkpoints"),
            resume_tag: None,
            use_fp16: true,
            seed: 42,
            no_augment: false,
            num_data_workers: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType { Arc, Sudoku, Maze }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsWeightScheme { Uniform, LinearIncreasing, LastOnly }

// ─── Async Data Loader ──────────────────────────────────────────────────

pub struct AsyncDataMessage {
    pub examples: Vec<TokenizedExample>,
    pub max_seq: usize,
    pub x: Tensor,
    pub y: Tensor,
    pub p: Tensor,
    pub mask: Tensor,
}

pub struct AsyncDataLoader;

impl AsyncDataLoader {
    pub fn start(
        train_data: Vec<TokenizedExample>,
        config: TrainConfig,
        device: Device,
        pad_token: u32,
        task_type: TaskType,
        rng_seed: u64,
    ) -> Receiver<AsyncDataMessage> {
        let (tx, rx) = bounded(2);

        std::thread::spawn(move || {
            let pipeline = AugmentPipeline {
                n_augmentations: config.n_augmentations_train,
                seed: config.aug_seed,
                parallel: true,
            };
            let cache = AugmentCache::new(config.aug_seed, config.n_augmentations_train);
            let mut rng = StdRng::seed_from_u64(rng_seed);
            let micro_batch = config.micro_batch_size;

            let mut indices: Vec<usize> = (0..train_data.len()).collect();
            indices.shuffle(&mut rng);

            for chunk in indices.chunks(micro_batch) {
                    let mut examples = Vec::with_capacity(chunk.len());

                    for &i in chunk {
                        let ex = &train_data[i];
                        if config.no_augment {
                            examples.push(ex.clone());
                        } else if task_type == TaskType::Sudoku {
                            let x_grid = detokenize_grid(&ex.x_tokens, ex.rows, ex.cols);
                            let y_grid = detokenize_grid(&ex.y_tokens, ex.rows, ex.cols);
                            let (x_aug, y_aug) = pipeline.sudoku_permute_pair(&x_grid, &y_grid);
                            let mut aug_ex = ex.clone();
                            aug_ex.x_tokens = tokenize_grid(&x_aug);
                            aug_ex.y_tokens = tokenize_grid(&y_aug);
                            aug_ex.x_seq_len = aug_ex.x_tokens.len();
                            aug_ex.y_seq_len = aug_ex.y_tokens.len();
                            examples.push(aug_ex);
                        } else {
                            let x_grid = detokenize_grid(&ex.x_tokens, ex.rows, ex.cols);
                            let y_grid = detokenize_grid(&ex.y_tokens, ex.rows, ex.cols);
                            let cached = cache.get_or_generate(ex.puzzle_id, &x_grid, &y_grid);
                            let (x_aug, y_aug) = cached.choose(&mut rng).unwrap().clone();
                            let mut aug_ex = ex.clone();
                            aug_ex.x_tokens = tokenize_grid(&x_aug);
                            aug_ex.y_tokens = tokenize_grid(&y_aug);
                            aug_ex.x_seq_len = aug_ex.x_tokens.len();
                            aug_ex.y_seq_len = aug_ex.y_tokens.len();
                            examples.push(aug_ex);
                        }
                    }

                    if examples.is_empty() { continue; }

                    let max_seq = examples.iter().map(|e| e.x_tokens.len()).max().unwrap_or(1);
                    let b = examples.len();
                    let pad = pad_token;
                    let mut x_flat = Vec::with_capacity(b * max_seq);
                    let mut y_flat = Vec::with_capacity(b * max_seq);
                    let mut pids = Vec::with_capacity(b);
                    let mut lengths = Vec::with_capacity(b);

                    for ex in &examples {
                        let x = pad_tokens(&ex.x_tokens, max_seq, pad);
                        let y = pad_tokens(&ex.y_tokens, max_seq, pad);
                        x_flat.extend(x);
                        y_flat.extend(y);
                        pids.push(ex.puzzle_id);
                        lengths.push(ex.x_seq_len);
                    }

                    let mut mask_flat = vec![0f32; b * max_seq * max_seq];
                    for (bi, &len) in lengths.iter().enumerate() {
                        for s1 in 0..max_seq {
                            for s2 in 0..max_seq {
                                let idx = (bi * max_seq + s1) * max_seq + s2;
                                if s1 >= len || s2 >= len {
                                    mask_flat[idx] = -f32::INFINITY;
                                }
                            }
                        }
                    }

                    let x = Tensor::from_vec(x_flat, (b, max_seq), &device)
                        .unwrap().to_dtype(DType::U32).unwrap();
                    let y = Tensor::from_vec(y_flat, (b, max_seq), &device)
                        .unwrap().to_dtype(DType::U32).unwrap();
                    let p = Tensor::from_vec(pids, b, &device)
                        .unwrap().to_dtype(DType::U32).unwrap();
                    let mask = Tensor::from_vec(mask_flat, (b, 1, max_seq, max_seq), &device)
                        .unwrap();

                    if tx.send(AsyncDataMessage { examples, max_seq, x, y, p, mask }).is_err() {
                        break;
                    }
            }
        });

        rx
    }
}

// ─── Trainer ─────────────────────────────────────────────────────────────

pub struct Trainer {
    pub model: TrmModel,
    pub optimizer: MultiGroupOptimizer,
    pub ema: EMA,
    pub scheduler: CosineScheduler,
    pub config: TrainConfig,
    pub meta: TrainingMeta,
    pub rng: StdRng,
    pub current_batch_size: usize,
    pub metrics: TrainMetrics,
}

impl Trainer {
    pub fn new(device: Device, net_cfg: NetworkConfig, trm_cfg: TrmConfig, config: TrainConfig) -> Result<Self> {
        let model = TrmModel::new(device, net_cfg, trm_cfg)?;
        let optimizer = MultiGroupOptimizer::trm_default(
            config.base_lr,
            config.puzzle_lr,
            config.weight_decay,
            config.puzzle_wd,
        );
        let ema = EMA::new(config.ema_decay, config.ema_warmup);
        let total_steps = config.max_epochs as u64 * 100;
        let scheduler = CosineScheduler::new(
            vec![config.base_lr, config.puzzle_lr],
            config.warmup_steps,
            total_steps,
        );
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            model,
            optimizer,
            ema,
            scheduler,
            config: config.clone(),
            meta: TrainingMeta::default(),
            rng,
            current_batch_size: config.batch_size,
            metrics: TrainMetrics::default(),
        })
    }

    pub fn resume(&mut self) -> Result<()> {
        if let Some(tag) = &self.config.resume_tag {
            self.meta = load_checkpoint(
                &self.config.checkpoint_dir,
                tag,
                &mut self.model.varmap,
                &mut self.ema,
            )?;
            self.current_batch_size = self.meta.current_batch_size;
            log::info!(
                "Resumed from epoch {}, step {}, best_acc {:.4}, batch {}",
                self.meta.epoch,
                self.meta.global_step,
                self.meta.best_val_acc,
                self.current_batch_size,
            );
        }
        Ok(())
    }

    pub fn load_data(&self) -> Result<(Vec<TokenizedExample>, Vec<TokenizedExample>)> {
        match self.config.task_type {
            TaskType::Arc => {
                let tasks = load_arc_tasks(&self.config.data_dir)?;
                let n = tasks.len();
                let split = (n * 9) / 10;
                let train_tasks = &tasks[..split];
                let val_tasks = &tasks[split..];
                let train = arc_training_examples(train_tasks);
                let val = arc_training_examples(val_tasks);
                log::info!("ARC: {} train, {} val examples", train.len(), val.len());
                Ok((train, val))
            }
            TaskType::Sudoku => {
                let path = self.config.data_dir.join("sudoku.csv");
                let all = load_sudoku_txt(&path)?;
                let split = (all.len() * 9) / 10;
                let (train, val) = all.split_at(split);
                Ok((train.to_vec(), val.to_vec()))
            }
            TaskType::Maze => {
                let all = load_maze_dir(&self.config.data_dir)?;
                let split = (all.len() * 9) / 10;
                let (train, val) = all.split_at(split);
                Ok((train.to_vec(), val.to_vec()))
            }
        }
    }

    pub fn train(&mut self) -> Result<()> {
        log::info!("=======================================================");
        log::info!(" TRM-Omega v10.2 Training Session");
        log::info!(" params      : {:.2}M live / ~{:.2}M layer est", self.model.param_count() as f64 / 1e6, self.model.net_cfg.param_count_estimate() as f64 / 1e6);
        log::info!(" batch       : {} (effective)", self.current_batch_size);
        log::info!(" task        : {:?}", self.config.task_type);
        log::info!(" DEQ mode    : {}", self.model.trm_cfg.use_deq);
        log::info!("=======================================================");

        let (train_data, val_data) = self.load_data()?;
        if train_data.is_empty() {
            anyhow::bail!("No training data found in {}", self.config.data_dir.display());
        }

        let start_epoch = self.meta.epoch;

        for epoch in start_epoch..self.config.max_epochs {
            self.meta.epoch = epoch;
            let t0 = Instant::now();

            let avg_loss = self.train_epoch(&train_data)?;
            let elapsed = t0.elapsed().as_secs_f64();

            log::info!("Epoch {}/{}: loss={:.4} time={:.1}s", epoch + 1, self.config.max_epochs, avg_loss, elapsed);

            if (epoch + 1) % self.config.eval_every_epochs == 0 {
                let (em_acc, tok_acc, val_loss) = self.evaluate(&val_data)?;
                log::info!(" VAL: exact={:.2}% token={:.2}% loss={:.4}", em_acc * 100.0, tok_acc * 100.0, val_loss);
                if em_acc > self.meta.best_val_acc {
                    self.meta.best_val_acc = em_acc;
                    save_checkpoint(&self.config.checkpoint_dir, "best", &self.model.varmap, &self.ema, &self.meta)?;
                }
            }

            if (epoch + 1) % self.config.save_every_epochs == 0 {
                self.meta.current_batch_size = self.current_batch_size;
                save_checkpoint(&self.config.checkpoint_dir, &format!("epoch_{}", epoch + 1), &self.model.varmap, &self.ema, &self.meta)?;
            }
        }

        save_checkpoint(&self.config.checkpoint_dir, "final", &self.model.varmap, &self.ema, &self.meta)?;
        Ok(())
    }

    pub fn train_epoch(&mut self, train_data: &[TokenizedExample]) -> Result<f64> {
        let rx = AsyncDataLoader::start(
            train_data.to_vec(),
            self.config.clone(),
            self.model.device.clone(),
            self.model.trm_cfg.blank_token_id,
            self.config.task_type,
            self.config.seed + self.meta.epoch as u64,
        );

        let mut epoch_loss = 0.0;
        let mut epoch_batches = 0;
        let mut grad_accum = 0.0;
        let mut step_count = 0;
        let accum_steps = (self.current_batch_size + self.config.micro_batch_size - 1)
            / self.config.micro_batch_size.max(1);
        let grad_scale = 1.0 / accum_steps.max(1) as f64;

        while let Ok(msg) = rx.recv() {
            let loss_val = self.process_batch(msg, grad_scale)?;
            grad_accum += loss_val;
            step_count += 1;

            if step_count % accum_steps == 0 {
                let lrs = self.scheduler.get_lrs(self.meta.global_step);
                self.optimizer.set_lrs(&lrs);
                let clip_scale = self.optimizer.clip_scale(self.config.grad_clip)?;
                self.optimizer.step(&self.model.varmap, clip_scale, 1.0)?;
                self.optimizer.zero_grad(&self.model.varmap)?;
                self.ema.update(&self.model.varmap)?;

                self.meta.global_step += 1;
                epoch_loss += grad_accum / accum_steps as f64;
                epoch_batches += 1;
                grad_accum = 0.0;
            }
        }

        Ok(epoch_loss / epoch_batches.max(1) as f64)
    }

    pub fn process_batch(&mut self, msg: AsyncDataMessage, grad_scale: f64) -> Result<f64> {
        let AsyncDataMessage { examples: _, max_seq, x, y, p, mask } = msg;

        let trace = self.model.forward_trace(
            &x,
            max_seq,
            Some(&p),
            Some(&y),
            None,
            Some(&mask),
        )?;

        let ds_weights = match self.config.ds_weight_scheme {
            DsWeightScheme::Uniform => None,
            DsWeightScheme::LinearIncreasing => {
                Some(linear_increasing_weights(trace.logits_intermediates.len()))
            }
            DsWeightScheme::LastOnly => {
                Some(crate::train::loss::last_step_only_weights(trace.logits_intermediates.len()))
            }
        };

        let ignore = Some(self.model.trm_cfg.blank_token_id);
        let ce_loss = deep_supervision_loss(
            &trace.logits_intermediates,
            &y,
            ds_weights.as_deref(),
            ignore,
        )?;

        let halt_loss = halting_ponder_loss(&trace.halt_probs, self.config.halt_lambda)?;
        let total_loss = (&ce_loss + &halt_loss)?;

        let loss_val = total_loss.to_scalar::<f32>()? as f64;

        if self.model.trm_cfg.use_deq {
            self.deq_backward(&x, Some(&p), max_seq, &y, &trace, Some(&mask), grad_scale)?;
        } else {
            let scaled = total_loss.affine(grad_scale, 0.0)?;
            let grads = scaled.backward()?;
            self.optimizer.accumulate(&self.model.varmap, &grads)?;
        }

        Ok(loss_val)
    }

    pub fn deq_backward(
        &mut self,
        x_tokens: &Tensor,
        puzzle_ids: Option<&Tensor>,
        y_seq: usize,
        targets: &Tensor,
        trace: &crate::recursion::TrmForwardTrace,
        mask: Option<&Tensor>,
        grad_scale: f64,
    ) -> Result<()> {
        let y_star = &trace.y_final;
        let z_star = &trace.z_final;
        let (batch, x_seq) = x_tokens.dims2()?;
        let z_seq = self.model.trm_cfg.z_seq;

        let x_emb = self.model.embed_question(x_tokens, puzzle_ids)?;
        let attn_mask = crate::recursion::expand_attn_mask(
            mask,
            batch,
            x_seq,
            y_seq,
            z_seq,
            &self.model.device,
        )?;
        let mask_owned = attn_mask;

        // Leaf variable for y* to obtain ∂L/∂y*
        let y_var = candle_core::Var::from_tensor(y_star)?;
        let logits = self.model.decode(y_var.as_tensor())?;
        let loss = cross_entropy(&logits, targets, Some(self.model.trm_cfg.blank_token_id))?
            .affine(grad_scale, 0.0)?;
        let grads = loss.backward()?;
        let grad_y = grads.get(y_var.as_tensor()).context("grad_y missing")?;
        let grad_z = Tensor::zeros_like(z_star)?;
        let grad_state = Tensor::cat(&[grad_y, &grad_z], 1)?;

        let fixed_state = Tensor::cat(&[y_star, z_star], 1)?;
        let deq = DeqWrapper::new(self.model.device.clone(), self.model.trm_cfg.deq_config.clone());

        let f_fn = |state: &Tensor| -> candle_core::Result<Tensor> {
            self.model.deq_transition(
                &x_emb,
                state,
                x_seq,
                y_seq,
                z_seq,
                mask_owned.as_ref(),
            )
        };

        let u = deq.backward(&fixed_state, &grad_state, f_fn)?;
        let f_val = f_fn(&fixed_state)?;
        let dot = (&u * &f_val)?.sum_all()?;
        let param_grads = dot.backward()?;
        self.optimizer.accumulate(&self.model.varmap, &param_grads)?;

        Ok(())
    }

    pub fn evaluate(&self, val_data: &[TokenizedExample]) -> Result<(f64, f64, f64)> {
        let originals = self.ema.swap_in(&self.model.varmap)?;
        let mut all_preds = Vec::new();
        let mut all_targets = Vec::new();
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        let batch_size = self.config.batch_size;
        let n = val_data.len();

        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let examples: Vec<_> = val_data[start..end].iter().collect();
            let max_seq = examples.iter().map(|e| e.x_tokens.len()).max().unwrap_or(1);
            let pad = self.model.trm_cfg.blank_token_id;

            let mut x_flat = Vec::with_capacity(examples.len() * max_seq);
            let mut y_flat = Vec::with_capacity(examples.len() * max_seq);
            let mut pids = Vec::with_capacity(examples.len());
            let mut lengths = Vec::with_capacity(examples.len());

            for ex in &examples {
                let x = pad_tokens(&ex.x_tokens, max_seq, pad);
                let y = pad_tokens(&ex.y_tokens, max_seq, pad);
                x_flat.extend(x);
                y_flat.extend(y);
                pids.push(ex.puzzle_id);
                lengths.push(ex.x_seq_len);
            }

            let b = examples.len();
            let device = &self.model.device;
            let x = Tensor::from_vec(x_flat, (b, max_seq), device)?.to_dtype(DType::U32)?;
            let y = Tensor::from_vec(y_flat, (b, max_seq), device)?.to_dtype(DType::U32)?;
            let p = Tensor::from_vec(pids, b, device)?.to_dtype(DType::U32)?;

            let mut mask_flat = vec![0f32; b * max_seq * max_seq];
            for (bi, &len) in lengths.iter().enumerate() {
                for s1 in 0..max_seq {
                    for s2 in 0..max_seq {
                        let idx = (bi * max_seq + s1) * max_seq + s2;
                        if s1 >= len || s2 >= len {
                            mask_flat[idx] = -f32::INFINITY;
                        }
                    }
                }
            }
            let mask = Tensor::from_vec(mask_flat, (b, 1, max_seq, max_seq), device)?;

            let trace = self.model.forward_trace(
                &x,
                max_seq,
                Some(&p),
                None,
                None,
                Some(&mask),
            )?;

            let loss = cross_entropy(&trace.logits_final, &y, Some(pad))?;
            total_loss += loss.to_scalar::<f32>()? as f64;
            n_batches += 1;

            let preds = trace.preds_final.to_vec2::<u32>()?;
            let targets: Vec<Vec<u32>> = examples.iter().map(|e| {
                pad_tokens(&e.y_tokens, max_seq, pad)
            }).collect();

            all_preds.extend(preds);
            all_targets.extend(targets);
        }

        let em_acc = exact_match_accuracy(&all_preds, &all_targets);
        let tok_acc = token_accuracy(&all_preds, &all_targets);
        let avg_loss = total_loss / n_batches.max(1) as f64;

        EMA::swap_out(&originals, &self.model.varmap)?;
        Ok((em_acc, tok_acc, avg_loss))
    }
}
