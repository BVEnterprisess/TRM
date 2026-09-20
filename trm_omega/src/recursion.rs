//! # TRM Recursion Core with DEQ Solver Integration
//!
//! Implements the exact TRM semantics with support for:
//! - Deep Equilibrium (DEQ) fixed-point solving with Anderson acceleration
//! - Unrolled autograd recurrence (L-cycles and H-cycles)
//! - Configurable detach between supervision steps
//! - Gradient checkpointing (only last K steps carry gradients)
//! - Learned halt head and adaptive halting
//! - Attention masking

use std::collections::BTreeMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{self as nn, Module, VarBuilder, VarMap};

use crate::deq::{DeqConfig, DeqWrapper};
use crate::kernel_dispatch::MaybeTernaryLinear;
use crate::network::{NetworkConfig, SharedNetwork, TrmEmbeddings};
use crate::quantize::TernaryPacked;

// ──────────────────────────────── Config ────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetachStrategy {
    AfterEachStep,
    NoDetach,
}

#[derive(Debug, Clone)]
pub struct TrmConfig {
    pub n_l_cycles: usize,
    pub n_sup: usize,
    pub halt_threshold: f32,
    pub use_adaptive_halt: bool,
    pub max_puzzles: usize,
    pub blank_token_id: u32,
    pub use_learned_halt_head: bool,
    pub collect_intermediates: bool,
    pub x_seq: usize,
    pub y_seq: usize,
    pub z_seq: usize,
    pub detach_strategy: DetachStrategy,
    pub n_gradient_sup: usize,
    pub normalize_z_between_l_cycles: bool,
    pub z_init_from_x_mean: bool,
    pub use_deq: bool,
    pub deq_config: DeqConfig,
}

impl Default for TrmConfig {
    fn default() -> Self {
        Self {
            n_l_cycles: 6,
            n_sup: 16,
            halt_threshold: 0.995,
            use_adaptive_halt: false,
            max_puzzles: 4096,
            blank_token_id: 10,
            use_learned_halt_head: true,
            collect_intermediates: true,
            x_seq: 64,
            y_seq: 32,
            z_seq: 81,
            detach_strategy: DetachStrategy::AfterEachStep,
            n_gradient_sup: 3,
            normalize_z_between_l_cycles: false,
            z_init_from_x_mean: false,
            use_deq: false,
            deq_config: DeqConfig::default(),
        }
    }
}

// ──────────────────────────────── Halt Head ────────────────────────────────

pub struct HaltHead {
    ln: nn::LayerNorm,
    fc1: MaybeTernaryLinear,
    fc2: MaybeTernaryLinear,
}

impl HaltHead {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            ln: nn::layer_norm(dim, 1e-5, vb.pp("ln"))?,
            fc1: MaybeTernaryLinear::new(nn::linear(dim, dim / 2, vb.pp("fc1"))?),
            fc2: MaybeTernaryLinear::new(nn::linear(dim / 2, 1, vb.pp("fc2"))?),
        })
    }

    pub fn forward(&self, y: &Tensor) -> Result<Tensor> {
        let (b, s, _) = y.dims3()?;
        let pooled = y.sum(1)?.affine(1.0 / s as f64, 0.0)?;
        let h = if self.fc1.is_packed() {
            crate::kernel_dispatch::layer_norm_module(&pooled, &self.ln)?
        } else {
            self.ln.forward(&pooled)?
        };
        let h = candle_nn::ops::silu(&self.fc1.forward(&h)?)?;
        let p = self.fc2.forward(&h)?;
        candle_nn::ops::sigmoid(&p)?.reshape((b,))
    }

    pub fn pack_from_weights(&mut self) -> anyhow::Result<()> {
        self.fc1.pack_from_weight()?;
        self.fc2.pack_from_weight()?;
        Ok(())
    }

    pub fn apply_ternary_packs(&mut self, prefix: &str, packs: &BTreeMap<String, TernaryPacked>) {
        self.fc1.apply_named(&format!("{prefix}.fc1"), packs);
        self.fc2.apply_named(&format!("{prefix}.fc2"), packs);
    }
}

// ──────────────────────────────── Main TRM Model ────────────────────────────────

pub struct TrmModel {
    pub device: Device,
    pub varmap: VarMap,
    pub net_cfg: NetworkConfig,
    pub trm_cfg: TrmConfig,
    pub embeddings: TrmEmbeddings,
    pub network: SharedNetwork,
    pub halt_head: Option<HaltHead>,
}

pub struct TrmForwardTrace {
    pub y_final: Tensor,
    pub z_final: Tensor,
    pub logits_final: Tensor,
    pub preds_final: Tensor,
    pub y_intermediates: Vec<Tensor>,
    pub logits_intermediates: Vec<Tensor>,
    pub halt_probs: Vec<Tensor>,
    pub steps_used: usize,
    pub grad_enabled_steps: usize,
}

/// Expand an x-token padding mask `(B, 1, x, x)` to the concatenated `[x|y|z]` length.
pub fn expand_attn_mask(
    mask: Option<&Tensor>,
    batch: usize,
    x_seq: usize,
    y_seq: usize,
    z_seq: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(m) = mask else {
        return Ok(None);
    };
    let total = x_seq + y_seq + z_seq;
    let last = *m.dims().last().unwrap_or(&0);
    if last == total {
        return Ok(Some(m.clone()));
    }
    if last != x_seq {
        candle_core::bail!(
            "attn mask last dim {last} matches neither x_seq {x_seq} nor concat {total}"
        );
    }
    let data = m.flatten_all()?.to_vec1::<f32>()?;
    let stride_b = data.len() / batch.max(1);
    let mut is_pad = vec![false; batch * x_seq];
    for b in 0..batch {
        for i in 0..x_seq {
            let diag = b * stride_b + i * x_seq + i;
            if diag < data.len() {
                is_pad[b * x_seq + i] = data[diag].is_infinite() && data[diag] < 0.0;
            }
        }
    }
    let mut full = vec![0f32; batch * total * total];
    for b in 0..batch {
        for q in 0..total {
            for k in 0..total {
                if k < x_seq && is_pad[b * x_seq + k] {
                    full[(b * total + q) * total + k] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Some(Tensor::from_vec(full, (batch, 1, total, total), device)?))
}

impl TrmModel {
    pub fn new(device: Device, net_cfg: NetworkConfig, trm_cfg: TrmConfig) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let embeddings = TrmEmbeddings::new(
            vb.pp("embeddings"),
            net_cfg.vocab_size,
            net_cfg.dim,
            trm_cfg.max_puzzles,
        )?;
        let network = SharedNetwork::new(vb.pp("shared_network"), net_cfg.clone(), &device)?;
        let halt_head = if trm_cfg.use_learned_halt_head {
            Some(HaltHead::new(vb.pp("halt_head"), net_cfg.dim)?)
        } else {
            None
        };
        Ok(Self { device, varmap, net_cfg, trm_cfg, embeddings, network, halt_head })
    }

    /// Quantize live 2D linears and route them through packed kernels (inference).
    pub fn pack_ternary_inference(&mut self) -> anyhow::Result<()> {
        self.network.pack_from_weights()?;
        if let Some(h) = &mut self.halt_head {
            h.pack_from_weights()?;
        }
        self.net_cfg.use_ternary_kernels = true;
        Ok(())
    }

    /// Attach TRMQ10 packed weights without re-quantizing (avoids alpha drift).
    pub fn apply_ternary_packs(&mut self, packs: &BTreeMap<String, TernaryPacked>) {
        self.network.apply_ternary_packs("shared_network", packs);
        if let Some(h) = &mut self.halt_head {
            h.apply_ternary_packs("halt_head", packs);
        }
        self.net_cfg.use_ternary_kernels = true;
    }

    pub fn param_count(&self) -> usize {
        let data = self.varmap.data().lock().unwrap();
        data.values().map(|v| v.as_tensor().elem_count()).sum()
    }

    pub fn packed_linear_count(&self) -> usize {
        let halt = self.halt_head.as_ref().map(|h| {
            usize::from(h.fc1.is_packed()) + usize::from(h.fc2.is_packed())
        }).unwrap_or(0);
        self.network.packed_linear_count() + halt
    }

    pub fn load_safetensors<P: AsRef<Path>>(
        device: Device,
        net_cfg: NetworkConfig,
        trm_cfg: TrmConfig,
        path: P,
    ) -> Result<Self> {
        let mut model = Self::new(device, net_cfg, trm_cfg)?;
        model.varmap.load(path)?;
        if model.net_cfg.use_ternary_kernels {
            model
                .pack_ternary_inference()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }
        Ok(model)
    }

    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path)
    }

    fn embed_segment(&self, seg_id: u32, batch: usize, seq: usize) -> Result<Tensor> {
        let ids = Tensor::from_vec(vec![seg_id; seq], seq, &self.device)?;
        let emb = self.embeddings.segment.forward(&ids)?.reshape((1, seq, self.net_cfg.dim))?;
        emb.broadcast_as((batch, seq, self.net_cfg.dim))
    }

    fn embed_puzzle(&self, puzzle_ids: Option<&Tensor>, batch: usize, seq: usize) -> Result<Option<Tensor>> {
        match puzzle_ids {
            Some(ids) => {
                let ids = ids.to_dtype(DType::U32)?;
                let pe = self.embeddings.puzzle.forward(&ids)?;
                let pe = pe.reshape((batch, 1, self.net_cfg.dim))?;
                Ok(Some(pe.broadcast_as((batch, seq, self.net_cfg.dim))?))
            }
            None => Ok(None),
        }
    }

    pub fn decode(&self, y: &Tensor) -> Result<Tensor> {
        let w = self.embeddings.decode_weight();
        let vocab = w.dims()[0];
        let wt = w.transpose(0, 1)?.contiguous()?;
        let (b, s, d) = y.dims3()?;
        y.reshape((b * s, d))?.matmul(&wt)?.reshape((b, s, vocab))
    }

    pub fn argmax_tokens(&self, logits: &Tensor) -> Result<Tensor> {
        logits.argmax(D::Minus1)
    }

    pub fn embed_question(
        &self,
        x_tokens: &Tensor,
        puzzle_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, x_seq) = x_tokens.dims2()?;
        let u32_tokens = x_tokens.to_dtype(DType::U32)?;
        let mut x = self.embeddings.token.forward(&u32_tokens)?;
        x = (&x + &self.embed_segment(0, b, x_seq)?)?;
        if let Some(pe) = self.embed_puzzle(puzzle_ids, b, x_seq)? {
            x = (&x + &pe)?;
        }
        Ok(x)
    }

    pub fn init_answer_and_latent(
        &self,
        batch: usize,
        y_seq: usize,
        z_seq: usize,
        puzzle_ids: Option<&Tensor>,
        y_seed_tokens: Option<&Tensor>,
        x_embed: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let d = self.net_cfg.dim;

        let mut y = self.embeddings.init_y.broadcast_as((batch, y_seq, d))?;
        let mut z = self.embeddings.init_z.broadcast_as((batch, z_seq, d))?;

        let seq_y = self.embed_segment(1, batch, y_seq)?;
        let seq_z = self.embed_segment(2, batch, z_seq)?;
        y = (&y + &seq_y)?;
        z = (&z + &seq_z)?;

        if let Some(seed) = y_seed_tokens {
            let seed = seed.to_dtype(DType::U32)?;
            let y_tok = self.embeddings.token.forward(&seed)?;
            y = (&y + &y_tok)?;
        } else {
            let blank = Tensor::from_vec(
                vec![self.trm_cfg.blank_token_id; batch * y_seq],
                (batch, y_seq),
                &self.device,
            )?;
            let y_tok = self.embeddings.token.forward(&blank)?;
            y = (&y + &y_tok)?;
        }

        if let Some(pe) = self.embed_puzzle(puzzle_ids, batch, y_seq)? {
            y = (&y + &pe)?;
        }
        if let Some(pe) = self.embed_puzzle(puzzle_ids, batch, z_seq)? {
            z = (&z + &pe)?;
        }

        if self.trm_cfg.z_init_from_x_mean {
            if let Some(x_emb) = x_embed {
                let mean = x_emb.mean(1)?;
                let mean = mean.reshape((batch, 1, d))?;
                let mean_bc = mean.broadcast_as((batch, z_seq, d))?;
                z = (&z + &mean_bc)?;
            }
        }

        Ok((y, z))
    }

    pub fn concat_xyz(&self, x: &Tensor, y: &Tensor, z: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[x, y, z], 1)
    }

    pub fn split_output(
        &self,
        output: &Tensor,
        x_seq: usize,
        y_seq: usize,
        z_seq: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x_out = output.narrow(1, 0, x_seq)?;
        let y_out = output.narrow(1, x_seq, y_seq)?;
        let z_out = output.narrow(1, x_seq + y_seq, z_seq)?;
        Ok((x_out, y_out, z_out))
    }

    fn heuristic_halt_confidence(&self, prev_y: &Tensor, curr_y: &Tensor) -> Result<Tensor> {
        let (b, y_seq, d) = curr_y.dims3()?;
        let prev = prev_y.flatten_all()?.to_vec1::<f32>()?;
        let curr = curr_y.flatten_all()?.to_vec1::<f32>()?;
        let elem = y_seq * d;
        let mut conf = vec![0f32; b];
        for bi in 0..b {
            let off = bi * elem;
            let old = &prev[off..off + elem];
            let new = &curr[off..off + elem];
            let mut delta = 0.0f32;
            let mut mag = 0.0f32;
            for i in 0..elem {
                delta += (new[i] - old[i]).abs();
                mag += new[i].abs();
            }
            conf[bi] = 1.0 - (delta / (mag + 1e-7)).min(1.0);
        }
        Tensor::from_vec(conf, b, &self.device)
    }

    /// One network pass writing both y and z. Kept for unit tests and kernels.
    pub fn recurse_step(
        &self,
        x: &Tensor,
        y: &Tensor,
        z: &Tensor,
        x_seq: usize,
        y_seq: usize,
        z_seq: usize,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let concat = self.concat_xyz(x, y, z)?;
        let out = self.network.forward(&concat, mask)?;
        let (_, y_new, z_new) = self.split_output(&out, x_seq, y_seq, z_seq)?;
        Ok((y_new, z_new))
    }

    /// One H-cycle: `n_l` z-only updates, then one y update. This is `f` for DEQ.
    /// Matches the unrolled inner body so `--deq` and unroll share the same operator.
    pub fn h_cycle_step(
        &self,
        x: &Tensor,
        y: &Tensor,
        z: &Tensor,
        x_seq: usize,
        y_seq: usize,
        z_seq: usize,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let mut z_cur = z.clone();
        for _ in 0..self.trm_cfg.n_l_cycles {
            let concat = self.concat_xyz(x, y, &z_cur)?;
            let out = self.network.forward(&concat, mask)?;
            let (_, _, z_new) = self.split_output(&out, x_seq, y_seq, z_seq)?;
            z_cur = z_new;
        }
        let concat = self.concat_xyz(x, y, &z_cur)?;
        let out = self.network.forward(&concat, mask)?;
        let (_, y_new, _) = self.split_output(&out, x_seq, y_seq, z_seq)?;
        Ok((y_new, z_cur))
    }

    pub fn deq_transition(
        &self,
        x: &Tensor,
        state: &Tensor,
        x_seq: usize,
        y_seq: usize,
        z_seq: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let y_cur = state.narrow(1, 0, y_seq)?;
        let z_cur = state.narrow(1, y_seq, z_seq)?;
        let (y_next, z_next) = self.h_cycle_step(x, &y_cur, &z_cur, x_seq, y_seq, z_seq, mask)?;
        Tensor::cat(&[&y_next, &z_next], 1)
    }

    // ─── Main Forward with DEQ + Mask Support ─────────────────────────

    pub fn forward_trace(
        &self,
        x_tokens: &Tensor,
        y_seq: usize,
        puzzle_ids: Option<&Tensor>,
        y_seed_tokens: Option<&Tensor>,
        max_sup_steps: Option<usize>,
        mask: Option<&Tensor>,
    ) -> Result<TrmForwardTrace> {
        let device = &self.device;
        let (batch, x_seq) = x_tokens.dims2()?;
        let z_seq = self.trm_cfg.z_seq;
        let n_sup = max_sup_steps.unwrap_or(self.trm_cfg.n_sup);

        let x = self.embed_question(x_tokens, puzzle_ids)?;
        let x_for_init = if self.trm_cfg.z_init_from_x_mean { Some(&x) } else { None };
        let (mut y, mut z) = self.init_answer_and_latent(
            batch,
            y_seq,
            z_seq,
            puzzle_ids,
            y_seed_tokens,
            x_for_init,
        )?;

        let attn_mask = expand_attn_mask(mask, batch, x_seq, y_seq, z_seq, device)?;
        let mask = attn_mask.as_ref();

        let mut y_intermediates = Vec::with_capacity(n_sup);
        let mut logits_intermediates = Vec::with_capacity(n_sup);
        let mut halt_probs = Vec::with_capacity(n_sup);
        let mut prev_y = y.clone();
        let mut steps_used = 0;
        let mut grad_enabled_steps = 0;

        // ─── DEQ Path: fixed point of one H-cycle ────────────────────
        if self.trm_cfg.use_deq {
            let deq = DeqWrapper::new(self.device.clone(), self.trm_cfg.deq_config.clone());
            let initial_state = Tensor::cat(&[&y, &z], 1)?;
            let x_ref = &x;
            let f_fn = |state: &Tensor| -> Result<Tensor> {
                self.deq_transition(x_ref, state, x_seq, y_seq, z_seq, mask)
            };

            let fixed_state = deq.forward(&initial_state, f_fn)?;
            y = fixed_state.narrow(1, 0, y_seq)?;
            z = fixed_state.narrow(1, y_seq, z_seq)?;

            let logits_final = self.decode(&y)?;
            let preds_final = self.argmax_tokens(&logits_final)?;
            if self.trm_cfg.collect_intermediates {
                logits_intermediates.push(logits_final.clone());
                y_intermediates.push(y.clone());
            }
            halt_probs.push(Tensor::ones(batch, DType::F32, device)?);

            return Ok(TrmForwardTrace {
                y_final: y,
                z_final: z,
                logits_final,
                preds_final,
                y_intermediates,
                logits_intermediates,
                halt_probs,
                steps_used: 1,
                grad_enabled_steps: 1,
            });
        }

        // ─── Standard Recurrent Unrolled Path ────────────────────────
        for sup in 0..n_sup {
            let carry_grad = sup >= n_sup.saturating_sub(self.trm_cfg.n_gradient_sup);
            if carry_grad {
                grad_enabled_steps += 1;
            }

            let (y_new, z_new) = self.h_cycle_step(&x, &y, &z, x_seq, y_seq, z_seq, mask)?;
            y = y_new;
            z = z_new;

            // Decode on the live graph so deep-supervision CE can train.
            if self.trm_cfg.collect_intermediates {
                let logits = self.decode(&y)?;
                logits_intermediates.push(logits);
                y_intermediates.push(y.clone());
            }

            let conf = if self.trm_cfg.use_adaptive_halt {
                if let Some(head) = &self.halt_head {
                    head.forward(&y)?
                } else {
                    self.heuristic_halt_confidence(&prev_y, &y)?
                }
            } else {
                Tensor::zeros(batch, DType::F32, device)?
            };
            halt_probs.push(conf.clone());

            if self.trm_cfg.use_adaptive_halt {
                let conf_vec = conf.to_vec1::<f32>()?;
                if conf_vec.iter().all(|&v| v >= self.trm_cfg.halt_threshold) {
                    steps_used = sup + 1;
                    break;
                }
            }

            if self.trm_cfg.detach_strategy == DetachStrategy::AfterEachStep && sup < n_sup - 1 {
                y = y.detach();
                z = z.detach();
                prev_y = y.clone();
            } else {
                prev_y = y.clone();
            }
        }

        if steps_used == 0 { steps_used = n_sup; }

        let logits_final = self.decode(&y)?;
        let preds_final = self.argmax_tokens(&logits_final)?;

        Ok(TrmForwardTrace {
            y_final: y,
            z_final: z,
            logits_final,
            preds_final,
            y_intermediates,
            logits_intermediates,
            halt_probs,
            steps_used,
            grad_enabled_steps,
        })
    }

    pub fn forward_inference(
        &self,
        x_tokens: &Tensor,
        y_seq: usize,
        puzzle_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let trace = self.forward_trace(x_tokens, y_seq, puzzle_ids, None, None, mask)?;
        Ok(trace.preds_final)
    }

    pub fn load_named_tensors(&self) -> BTreeMap<String, Tensor> {
        let data = self.varmap.data().lock().unwrap();
        data.iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
            .collect()
    }
}
