//! # Shared Network f_0
//!
//! The single network applied recursively for ALL L-cycles and H-cycles.
//! Built on candle-nn for autograd support.

use std::collections::BTreeMap;

use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{self as nn, Embedding, LayerNorm as CLayerNorm, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::kernel_dispatch::MaybeTernaryLinear;
use crate::quantize::TernaryPacked;

// ─── Positional Encoding ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PositionalEncodingType {
    Rope,
    TwoDGrid,
    Learned1D,
    None,
}

// ─── Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NetworkVariant {
    Transformer,
    MlpMixer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub variant: NetworkVariant,
    pub dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>,
    pub ffn_mult: f64,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub dropout_p: f64,
    pub rope_base: f64,
    pub tie_embeddings: bool,
    pub pos_enc: PositionalEncodingType,
    pub allow_non_standard_depth: bool,
    pub use_inplace_layernorm: bool,
    /// When true, 2D linears with `in_dim % 4 == 0` run through packed ternary kernels.
    #[serde(default)]
    pub use_ternary_kernels: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            variant: NetworkVariant::Transformer,
            dim: 256,
            num_heads: 8,
            num_kv_heads: None,
            ffn_mult: 8.0 / 3.0,
            max_seq_len: 900,
            num_layers: 2,
            vocab_size: 11,
            dropout_p: 0.0,
            rope_base: 10_000.0,
            tie_embeddings: true,
            pos_enc: PositionalEncodingType::Rope,
            allow_non_standard_depth: false,
            use_inplace_layernorm: true,
            use_ternary_kernels: false,
        }
    }
}

impl NetworkConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.allow_non_standard_depth && self.num_layers != 2 {
            candle_core::bail!(
                "TRM paper found 2 layers optimal; got {}. Set allow_non_standard_depth=true.",
                self.num_layers
            );
        }
        if self.dim % self.num_heads != 0 {
            candle_core::bail!("dim must be divisible by num_heads");
        }
        if let Some(kv) = self.num_kv_heads {
            if self.num_heads % kv != 0 {
                candle_core::bail!("num_heads must be divisible by num_kv_heads");
            }
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize { self.dim / self.num_heads }
    pub fn ffn_hidden(&self) -> usize {
        let raw = (self.dim as f64 * self.ffn_mult).round() as usize;
        (raw + 7) & !7
    }
    pub fn kv_heads(&self) -> usize { self.num_kv_heads.unwrap_or(self.num_heads) }

    pub fn param_count_estimate(&self) -> usize {
        let d = self.dim;
        let h = self.ffn_hidden();
        match self.variant {
            NetworkVariant::Transformer => {
                let per_layer = 4 * d * d + 3 * d * h + 4 * d;
                per_layer * self.num_layers
            }
            NetworkVariant::MlpMixer => {
                let s = self.max_seq_len;
                let per_layer = 3 * s * s + 3 * d * h + 4 * d;
                per_layer * self.num_layers
            }
        }
    }
}

// ─── RoPE ─────────────────────────────────────────────────────────

fn build_rope_cache(device: &Device, max_seq: usize, head_dim: usize, base: f64) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = vec![0f32; max_seq * half];
    let mut sin = vec![0f32; max_seq * half];
    for pos in 0..max_seq {
        for i in 0..half {
            let theta = (pos as f64) / base.powf((2 * i) as f64 / head_dim as f64);
            cos[pos * half + i] = theta.cos() as f32;
            sin[pos * half + i] = theta.sin() as f32;
        }
    }
    let cos = Tensor::from_vec(cos, (max_seq, half), device)?;
    let sin = Tensor::from_vec(sin, (max_seq, half), device)?;
    Ok((cos, sin))
}

fn apply_rope(q_or_k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, s, dh) = q_or_k.dims4()?;
    let half = dh / 2;
    let x1 = q_or_k.narrow(D::Minus1, 0, half)?;
    let x2 = q_or_k.narrow(D::Minus1, half, half)?;
    let cos = cos.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
    let sin = sin.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
    let x1c = x1.broadcast_mul(&cos)?;
    let x2s = x2.broadcast_mul(&sin)?;
    let x1s = x1.broadcast_mul(&sin)?;
    let x2c = x2.broadcast_mul(&cos)?;
    let rot1 = (&x1c - &x2s)?;
    let rot2 = (&x1s + &x2c)?;
    Tensor::cat(&[&rot1, &rot2], D::Minus1)
}

// ─── 2D Grid Positional Encoding ────────────────────────────────

pub struct TwoDGridEmbedding {
    row_emb: Embedding,
    col_emb: Embedding,
    dim: usize,
}

impl TwoDGridEmbedding {
    pub fn new(vb: VarBuilder, max_rows: usize, max_cols: usize, dim: usize) -> Result<Self> {
        let half = dim / 2;
        Ok(Self {
            row_emb: nn::embedding(max_rows, half, vb.pp("row_emb"))?,
            col_emb: nn::embedding(max_cols, half, vb.pp("col_emb"))?,
            dim,
        })
    }
    pub fn forward(&self, rows: &Tensor, cols: &Tensor) -> Result<Tensor> {
        let r = self.row_emb.forward(rows)?;
        let c = self.col_emb.forward(cols)?;
        Tensor::cat(&[&r, &c], D::Minus1)
    }
}

// ─── SwiGLU FFN ──────────────────────────────────────────────────

pub struct SwiGluFfn {
    gate: MaybeTernaryLinear,
    value: MaybeTernaryLinear,
    down: MaybeTernaryLinear,
}

impl SwiGluFfn {
    pub fn new(vb: VarBuilder, dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            gate: MaybeTernaryLinear::new(nn::linear(dim, hidden, vb.pp("gate"))?),
            value: MaybeTernaryLinear::new(nn::linear(dim, hidden, vb.pp("value"))?),
            down: MaybeTernaryLinear::new(nn::linear(hidden, dim, vb.pp("down"))?),
        })
    }
    pub fn is_packed(&self) -> bool {
        self.gate.is_packed() && self.value.is_packed() && self.down.is_packed()
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if self.is_packed() {
            return crate::kernel_dispatch::packed_swiglu_ffn(xs, &self.gate, &self.value, &self.down);
        }
        let gate = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        let value = self.value.forward(xs)?;
        let hidden = gate.broadcast_mul(&value)?;
        self.down.forward(&hidden)
    }
    pub fn pack_from_weights(&mut self) -> anyhow::Result<()> {
        self.gate.pack_from_weight()?;
        self.value.pack_from_weight()?;
        self.down.pack_from_weight()?;
        Ok(())
    }
    pub fn apply_ternary_packs(&mut self, prefix: &str, packs: &BTreeMap<String, TernaryPacked>) {
        self.gate.apply_named(&format!("{prefix}.gate"), packs);
        self.value.apply_named(&format!("{prefix}.value"), packs);
        self.down.apply_named(&format!("{prefix}.down"), packs);
    }
}

// ─── Multi-Head Attention (with GQA & Masking) ──────────────────

pub struct MultiHeadSelfAttention {
    q_proj: MaybeTernaryLinear,
    k_proj: MaybeTernaryLinear,
    v_proj: MaybeTernaryLinear,
    o_proj: MaybeTernaryLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    pub fn new(vb: VarBuilder, cfg: &NetworkConfig) -> Result<Self> {
        let kv = cfg.kv_heads();
        Ok(Self {
            q_proj: MaybeTernaryLinear::new(nn::linear(cfg.dim, cfg.dim, vb.pp("q_proj"))?),
            k_proj: MaybeTernaryLinear::new(nn::linear(cfg.dim, kv * cfg.head_dim(), vb.pp("k_proj"))?),
            v_proj: MaybeTernaryLinear::new(nn::linear(cfg.dim, kv * cfg.head_dim(), vb.pp("v_proj"))?),
            o_proj: MaybeTernaryLinear::new(nn::linear(cfg.dim, cfg.dim, vb.pp("o_proj"))?),
            n_heads: cfg.num_heads,
            n_kv_heads: kv,
            head_dim: cfg.head_dim(),
        })
    }

    pub fn is_packed(&self) -> bool {
        self.q_proj.is_packed() && self.k_proj.is_packed() && self.v_proj.is_packed() && self.o_proj.is_packed()
    }

    pub fn pack_from_weights(&mut self) -> anyhow::Result<()> {
        self.q_proj.pack_from_weight()?;
        self.k_proj.pack_from_weight()?;
        self.v_proj.pack_from_weight()?;
        self.o_proj.pack_from_weight()?;
        Ok(())
    }

    pub fn apply_ternary_packs(&mut self, prefix: &str, packs: &BTreeMap<String, TernaryPacked>) {
        self.q_proj.apply_named(&format!("{prefix}.q_proj"), packs);
        self.k_proj.apply_named(&format!("{prefix}.k_proj"), packs);
        self.v_proj.apply_named(&format!("{prefix}.v_proj"), packs);
        self.o_proj.apply_named(&format!("{prefix}.o_proj"), packs);
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.is_packed() {
            return crate::kernel_dispatch::packed_mha(
                xs,
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                &self.o_proj,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                rope_cos,
                rope_sin,
                mask,
            );
        }
        let (b, s, d) = xs.dims3()?;
        let h = self.n_heads;
        let kv = self.n_kv_heads;
        let dh = self.head_dim;

        let q = self.q_proj.forward(xs)?.reshape((b, s, h, dh))?.transpose(1, 2)?;
        let k = self.k_proj.forward(xs)?.reshape((b, s, kv, dh))?.transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?.reshape((b, s, kv, dh))?.transpose(1, 2)?;

        let q = apply_rope(&q, rope_cos, rope_sin)?.contiguous()?;
        let k = apply_rope(&k, rope_cos, rope_sin)?.contiguous()?;

        let k = if kv < h {
            let repeat = h / kv;
            k.repeat((1, repeat, 1, 1))?.contiguous()?
        } else {
            k
        };
        let v = if kv < h {
            let repeat = h / kv;
            v.repeat((1, repeat, 1, 1))?.contiguous()?
        } else {
            v.contiguous()?
        };

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;
        let scores = scores.affine(1.0 / (dh as f64).sqrt(), 0.0)?;

        let scores = if let Some(mask) = mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = attn.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, s, d))?;
        self.o_proj.forward(&ctx)
    }
}

// ─── Transformer Block ───────────────────────────────────────────

pub struct TransformerBlock {
    ln_attn: CLayerNorm,
    attn: MultiHeadSelfAttention,
    ln_ffn: CLayerNorm,
    ffn: SwiGluFfn,
    use_inplace: bool,
}

fn ln_vecs(ln: &CLayerNorm) -> Result<(Vec<f32>, Vec<f32>)> {
    let g = ln.weight().flatten_all()?.to_vec1::<f32>()?;
    let b = match ln.bias() {
        Some(bias) => bias.flatten_all()?.to_vec1::<f32>()?,
        None => vec![0f32; g.len()],
    };
    Ok((g, b))
}

fn linear_bias_vec(lin: &MaybeTernaryLinear) -> Result<Option<Vec<f32>>> {
    match lin.inner().bias() {
        Some(b) => Ok(Some(b.flatten_all()?.to_vec1::<f32>()?)),
        None => Ok(None),
    }
}

impl TransformerBlock {
    pub fn is_packed(&self) -> bool {
        self.attn.is_packed() && self.ffn.is_packed()
    }

    pub(crate) fn packed_layer_bufs(&self) -> Result<crate::kernel_dispatch::PackedLayerBufs<'_>> {
        let q = self.attn.q_proj.packed().ok_or_else(|| candle_core::Error::Msg("q not packed".into()))?;
        let k = self.attn.k_proj.packed().ok_or_else(|| candle_core::Error::Msg("k not packed".into()))?;
        let v = self.attn.v_proj.packed().ok_or_else(|| candle_core::Error::Msg("v not packed".into()))?;
        let o = self.attn.o_proj.packed().ok_or_else(|| candle_core::Error::Msg("o not packed".into()))?;
        let gate = self.ffn.gate.packed().ok_or_else(|| candle_core::Error::Msg("gate not packed".into()))?;
        let value = self.ffn.value.packed().ok_or_else(|| candle_core::Error::Msg("value not packed".into()))?;
        let down = self.ffn.down.packed().ok_or_else(|| candle_core::Error::Msg("down not packed".into()))?;
        let (ln_attn_g, ln_attn_b) = ln_vecs(&self.ln_attn)?;
        let (ln_ffn_g, ln_ffn_b) = ln_vecs(&self.ln_ffn)?;
        Ok(crate::kernel_dispatch::PackedLayerBufs {
            ln_attn_g,
            ln_attn_b,
            ln_ffn_g,
            ln_ffn_b,
            q,
            q_bias: linear_bias_vec(&self.attn.q_proj)?,
            k,
            k_bias: linear_bias_vec(&self.attn.k_proj)?,
            v,
            v_bias: linear_bias_vec(&self.attn.v_proj)?,
            o,
            o_bias: linear_bias_vec(&self.attn.o_proj)?,
            gate,
            gate_bias: linear_bias_vec(&self.ffn.gate)?,
            value,
            value_bias: linear_bias_vec(&self.ffn.value)?,
            down,
            down_bias: linear_bias_vec(&self.ffn.down)?,
            n_heads: self.attn.n_heads,
            n_kv: self.attn.n_kv_heads,
            head_dim: self.attn.head_dim,
        })
    }

    pub fn new(vb: VarBuilder, cfg: &NetworkConfig, _device: &Device) -> Result<Self> {
        Ok(Self {
            ln_attn: nn::layer_norm(cfg.dim, 1e-5, vb.pp("ln_attn"))?,
            attn: MultiHeadSelfAttention::new(vb.pp("attn"), cfg)?,
            ln_ffn: nn::layer_norm(cfg.dim, 1e-5, vb.pp("ln_ffn"))?,
            ffn: SwiGluFfn::new(vb.pp("ffn"), cfg.dim, cfg.ffn_hidden())?,
            use_inplace: cfg.use_inplace_layernorm,
        })
    }

    pub fn forward(&self, xs: &Tensor, rope_cos: &Tensor, rope_sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let attn_in = if self.attn.is_packed() {
            crate::kernel_dispatch::layer_norm_module(xs, &self.ln_attn)?
        } else {
            self.ln_attn.forward(xs)?
        };
        let attn_out = self.attn.forward(&attn_in, rope_cos, rope_sin, mask)?;
        let x = (xs + &attn_out)?;
        let ffn_in = if self.ffn.is_packed() {
            crate::kernel_dispatch::layer_norm_module(&x, &self.ln_ffn)?
        } else {
            self.ln_ffn.forward(&x)?
        };
        let ffn_out = self.ffn.forward(&ffn_in)?;
        (&x + &ffn_out)
    }

    pub fn pack_from_weights(&mut self) -> anyhow::Result<()> {
        self.attn.pack_from_weights()?;
        self.ffn.pack_from_weights()?;
        Ok(())
    }

    pub fn apply_ternary_packs(&mut self, prefix: &str, packs: &BTreeMap<String, TernaryPacked>) {
        self.attn.apply_ternary_packs(&format!("{prefix}.attn"), packs);
        self.ffn.apply_ternary_packs(&format!("{prefix}.ffn"), packs);
    }
}

// ─── MLP-Mixer Block ────────────────────────────────────────────

pub struct MixerBlock {
    ln_token: CLayerNorm,
    token_ffn: SwiGluFfn,
    ln_channel: CLayerNorm,
    channel_ffn: SwiGluFfn,
    seq_len: usize,
    use_inplace: bool,
}

impl MixerBlock {
    pub fn new(vb: VarBuilder, cfg: &NetworkConfig) -> Result<Self> {
        let s = cfg.max_seq_len;
        let token_hidden = s * 2;
        Ok(Self {
            ln_token: nn::layer_norm(cfg.dim, 1e-5, vb.pp("ln_token"))?,
            token_ffn: SwiGluFfn::new(vb.pp("token_ffn"), s, token_hidden)?,
            ln_channel: nn::layer_norm(cfg.dim, 1e-5, vb.pp("ln_channel"))?,
            channel_ffn: SwiGluFfn::new(vb.pp("channel_ffn"), cfg.dim, cfg.ffn_hidden())?,
            seq_len: s,
            use_inplace: cfg.use_inplace_layernorm,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, d) = xs.dims3()?;
        if s != self.seq_len {
            candle_core::bail!("MLP-Mixer requires seq_len={} got {}", self.seq_len, s);
        }

        let tm_in = if self.token_ffn.is_packed() {
            crate::kernel_dispatch::layer_norm_module(xs, &self.ln_token)?
        } else {
            self.ln_token.forward(xs)?
        };
        let tm_in = tm_in.transpose(1, 2)?.reshape((b * d, s))?;
        let tm_out = self.token_ffn.forward(&tm_in)?;
        let tm_out = tm_out.reshape((b, d, s))?.transpose(1, 2)?;
        let x = (xs + &tm_out)?;

        let cm_in = if self.channel_ffn.is_packed() {
            crate::kernel_dispatch::layer_norm_module(&x, &self.ln_channel)?
        } else {
            self.ln_channel.forward(&x)?
        };
        let cm_out = self.channel_ffn.forward(&cm_in)?;
        (&x + &cm_out)
    }

    pub fn pack_from_weights(&mut self) -> anyhow::Result<()> {
        self.token_ffn.pack_from_weights()?;
        self.channel_ffn.pack_from_weights()?;
        Ok(())
    }

    pub fn apply_ternary_packs(&mut self, prefix: &str, packs: &BTreeMap<String, TernaryPacked>) {
        self.token_ffn.apply_ternary_packs(&format!("{prefix}.token_ffn"), packs);
        self.channel_ffn.apply_ternary_packs(&format!("{prefix}.channel_ffn"), packs);
    }
}

// ─── Shared Network ──────────────────────────────────────────────

enum Block {
    Transformer(TransformerBlock),
    Mixer(MixerBlock),
}

pub struct SharedNetwork {
    cfg: NetworkConfig,
    blocks: Vec<Block>,
    rope_cos: Option<Tensor>,
    rope_sin: Option<Tensor>,
}

impl SharedNetwork {
    pub fn new(vb: VarBuilder, cfg: NetworkConfig, device: &Device) -> Result<Self> {
        cfg.validate()?;
        let est = cfg.param_count_estimate() as f64 / 1e6;
        log::info!("Estimated parameters: {:.2}M", est);

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        match cfg.variant {
            NetworkVariant::Transformer => {
                for i in 0..cfg.num_layers {
                    blocks.push(Block::Transformer(TransformerBlock::new(
                        vb.pp(format!("blocks.{i}")),
                        &cfg,
                        device,
                    )?));
                }
                let (cos, sin) = build_rope_cache(device, cfg.max_seq_len, cfg.head_dim(), cfg.rope_base)?;
                Ok(Self { cfg, blocks, rope_cos: Some(cos), rope_sin: Some(sin) })
            }
            NetworkVariant::MlpMixer => {
                for i in 0..cfg.num_layers {
                    blocks.push(Block::Mixer(MixerBlock::new(
                        vb.pp(format!("blocks.{i}")),
                        &cfg,
                    )?));
                }
                Ok(Self { cfg, blocks, rope_cos: None, rope_sin: None })
            }
        }
    }

    pub fn config(&self) -> &NetworkConfig { &self.cfg }

    pub fn pack_from_weights(&mut self) -> anyhow::Result<()> {
        for block in &mut self.blocks {
            match block {
                Block::Transformer(b) => b.pack_from_weights()?,
                Block::Mixer(b) => b.pack_from_weights()?,
            }
        }
        self.cfg.use_ternary_kernels = true;
        Ok(())
    }

    pub fn apply_ternary_packs(&mut self, prefix: &str, packs: &BTreeMap<String, TernaryPacked>) {
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let p = format!("{prefix}.blocks.{i}");
            match block {
                Block::Transformer(b) => b.apply_ternary_packs(&p, packs),
                Block::Mixer(b) => b.apply_ternary_packs(&p, packs),
            }
        }
        self.cfg.use_ternary_kernels = true;
    }

    pub fn packed_linear_count(&self) -> usize {
        fn count_ffn(f: &SwiGluFfn) -> usize {
            [&f.gate, &f.value, &f.down].iter().filter(|l| l.is_packed()).count()
        }
        self.blocks.iter().map(|b| match b {
            Block::Transformer(tb) => {
                let attn = [&tb.attn.q_proj, &tb.attn.k_proj, &tb.attn.v_proj, &tb.attn.o_proj]
                    .iter()
                    .filter(|l| l.is_packed())
                    .count();
                attn + count_ffn(&tb.ffn)
            }
            Block::Mixer(mb) => count_ffn(&mb.token_ffn) + count_ffn(&mb.channel_ffn),
        }).sum()
    }

    fn transformer_stack_packed(&self) -> bool {
        !self.blocks.is_empty()
            && self.cfg.variant == NetworkVariant::Transformer
            && self.blocks.iter().all(|b| match b {
                Block::Transformer(t) => t.is_packed(),
                Block::Mixer(_) => false,
            })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        if self.transformer_stack_packed() {
            let mut bufs = Vec::with_capacity(self.blocks.len());
            for block in &self.blocks {
                match block {
                    Block::Transformer(b) => bufs.push(b.packed_layer_bufs()?),
                    Block::Mixer(_) => unreachable!(),
                }
            }
            let rope_cos = self.rope_cos.as_ref().unwrap();
            let rope_sin = self.rope_sin.as_ref().unwrap();
            if let Ok(out) = crate::kernel_dispatch::packed_transformer_stack(
                xs, &bufs, rope_cos, rope_sin, mask,
            ) {
                return Ok(out);
            }
        }
        let mut h = xs.clone();
        match self.cfg.variant {
            NetworkVariant::Transformer => {
                let rope_cos = self.rope_cos.as_ref().unwrap();
                let rope_sin = self.rope_sin.as_ref().unwrap();
                for block in &self.blocks {
                    match block {
                        Block::Transformer(b) => {
                            h = b.forward(&h, rope_cos, rope_sin, mask)?;
                        }
                        _ => unreachable!(),
                    }
                }
            }
            NetworkVariant::MlpMixer => {
                for block in &self.blocks {
                    match block {
                        Block::Mixer(b) => { h = b.forward(&h)?; }
                        _ => unreachable!(),
                    }
                }
            }
        }
        Ok(h)
    }
}

// ─── Embeddings ──────────────────────────────────────────────────

pub struct TrmEmbeddings {
    pub token: Embedding,
    pub segment: Embedding,
    pub puzzle: Embedding,
    pub init_y: Tensor,
    pub init_z: Tensor,
    dim: usize,
    vocab_size: usize,
}

impl TrmEmbeddings {
    pub fn new(vb: VarBuilder, vocab_size: usize, dim: usize, max_puzzles: usize) -> Result<Self> {
        Ok(Self {
            token: nn::embedding(vocab_size, dim, vb.pp("token"))?,
            segment: nn::embedding(3, dim, vb.pp("segment"))?,
            puzzle: nn::embedding(max_puzzles, dim, vb.pp("puzzle"))?,
            init_y: vb.get((1, 1, dim), "init_y")?,
            init_z: vb.get((1, 1, dim), "init_z")?,
            dim,
            vocab_size,
        })
    }
    pub fn decode_weight(&self) -> Tensor {
        self.token.embeddings().clone()
    }
}
