//! Dispatch ternary GEMM: CUDA KernelBank when live, otherwise `kernel_ref`.
//!
//! Training stays on Candle autograd. Packed kernels are inference-only
//! (no gradient through the quantized weights).

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::Linear;

use crate::kernel_ref;
use crate::quantize::{tensor_to_ternary, TernaryPacked};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    CpuRef,
    #[cfg(feature = "cuda")]
    Cuda,
}

pub fn active_backend() -> KernelBackend {
    #[cfg(feature = "cuda")]
    {
        if crate::custom_kernels::global_bank().is_some() {
            return KernelBackend::Cuda;
        }
    }
    KernelBackend::CpuRef
}

/// KernelBank HtoD/DtoH counters. `None` when CUDA is off or the bank never started.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GpuCopyStats {
    pub htod_calls: u64,
    pub dtoh_calls: u64,
    pub htod_bytes: u64,
    pub dtoh_bytes: u64,
    pub htod_mb: f64,
    pub dtoh_mb: f64,
}

pub fn reset_copy_stats() {
    #[cfg(feature = "cuda")]
    if let Some(bank) = crate::custom_kernels::global_bank_if_ready() {
        bank.reset_copy_stats();
    }
}

pub fn snapshot_copy_stats() -> Option<GpuCopyStats> {
    #[cfg(feature = "cuda")]
    {
        return crate::custom_kernels::global_bank_if_ready().map(|bank| {
            let s = bank.snapshot_copy_stats();
            GpuCopyStats {
                htod_calls: s.htod_calls,
                dtoh_calls: s.dtoh_calls,
                htod_bytes: s.htod_bytes,
                dtoh_bytes: s.dtoh_bytes,
                htod_mb: s.htod_mb,
                dtoh_mb: s.dtoh_mb,
            }
        });
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

pub fn swiglu_f32(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if let Some(bank) = crate::custom_kernels::global_bank() {
            if bank.has("swiglu_fused") {
                if let Ok(out) = bank.swiglu_fused_host(gate, up) {
                    return Ok(out);
                }
            }
        }
    }
    Ok(kernel_ref::swiglu_fused(gate, up))
}

pub fn swiglu_tensors(gate: &Tensor, up: &Tensor) -> candle_core::Result<Tensor> {
    if gate.shape() != up.shape() {
        candle_core::bail!("swiglu shape mismatch {:?} vs {:?}", gate.shape(), up.shape());
    }
    let dims = gate.dims().to_vec();
    let g = gate.flatten_all()?.to_vec1::<f32>()?;
    let u = up.flatten_all()?.to_vec1::<f32>()?;
    let out = swiglu_f32(&g, &u)
        .map_err(|e| candle_core::Error::Msg(format!("swiglu kernel: {e:#}")))?;
    Tensor::from_vec(out, dims, gate.device())
}

pub fn ternary_linear_f32(input: &[f32], pack: &TernaryPacked, tokens: usize) -> Result<Vec<f32>> {
    packed_linear_f32(input, pack, None, tokens)
}

pub fn packed_linear_f32(
    input: &[f32],
    pack: &TernaryPacked,
    bias: Option<&[f32]>,
    tokens: usize,
) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if let Some(bank) = crate::custom_kernels::global_bank() {
            if let Ok(out) = bank.packed_linear_host(input, pack, bias, tokens) {
                return Ok(out);
            }
        }
    }
    let mut out = kernel_ref::ternary_matmul_stack(input, pack, tokens);
    if let Some(b) = bias {
        kernel_ref::bias_add(&mut out, b, tokens, pack.out_dim);
    }
    Ok(out)
}

pub fn packed_swiglu_ffn_f32(
    xs: &[f32],
    tokens: usize,
    gate: &TernaryPacked,
    gate_bias: Option<&[f32]>,
    value: &TernaryPacked,
    value_bias: Option<&[f32]>,
    down: &TernaryPacked,
    down_bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if let Some(bank) = crate::custom_kernels::global_bank() {
            if bank.has("swiglu_fused") {
                if let Ok(out) = bank.packed_swiglu_ffn_host(
                    xs, tokens, gate, gate_bias, value, value_bias, down, down_bias,
                ) {
                    return Ok(out);
                }
            }
        }
    }
    Ok(kernel_ref::packed_swiglu_ffn(
        xs, tokens, gate, gate_bias, value, value_bias, down, down_bias,
    ))
}

fn tensor_bias_vec(bias: Option<&Tensor>) -> candle_core::Result<Option<Vec<f32>>> {
    match bias {
        Some(b) => Ok(Some(b.flatten_all()?.to_vec1::<f32>()?)),
        None => Ok(None),
    }
}

/// Apply a packed ternary linear to a tensor whose last dim is `pack.in_dim`.
pub fn packed_linear(xs: &Tensor, pack: &TernaryPacked, bias: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let dims = xs.dims().to_vec();
    let in_dim = *dims.last().unwrap_or(&0);
    if in_dim != pack.in_dim {
        candle_core::bail!(
            "packed linear in_dim mismatch: tensor last dim {in_dim} vs pack {}",
            pack.in_dim
        );
    }
    let tokens = xs.elem_count() / in_dim.max(1);
    let input = xs.flatten_all()?.to_vec1::<f32>()?;
    let bias_v = tensor_bias_vec(bias)?;
    if let Some(b) = bias_v.as_ref() {
        if b.len() != pack.out_dim {
            candle_core::bail!("bias len {} != out_dim {}", b.len(), pack.out_dim);
        }
    }
    let out = packed_linear_f32(&input, pack, bias_v.as_deref(), tokens)
        .map_err(|e| candle_core::Error::Msg(format!("ternary kernel: {e:#}")))?;
    let mut out_dims = dims;
    if let Some(last) = out_dims.last_mut() {
        *last = pack.out_dim;
    }
    Tensor::from_vec(out, out_dims, xs.device())
}

pub fn packed_swiglu_ffn(
    xs: &Tensor,
    gate: &MaybeTernaryLinear,
    value: &MaybeTernaryLinear,
    down: &MaybeTernaryLinear,
) -> candle_core::Result<Tensor> {
    let gpack = gate.packed().ok_or_else(|| candle_core::Error::Msg("gate not packed".into()))?;
    let vpack = value.packed().ok_or_else(|| candle_core::Error::Msg("value not packed".into()))?;
    let dpack = down.packed().ok_or_else(|| candle_core::Error::Msg("down not packed".into()))?;
    let dims = xs.dims().to_vec();
    let in_dim = *dims.last().unwrap_or(&0);
    if in_dim != gpack.in_dim {
        candle_core::bail!("ffn in_dim mismatch: tensor {in_dim} vs gate pack {}", gpack.in_dim);
    }
    let tokens = xs.elem_count() / in_dim.max(1);
    let input = xs.flatten_all()?.to_vec1::<f32>()?;
    let gb = tensor_bias_vec(gate.inner().bias())?;
    let vb = tensor_bias_vec(value.inner().bias())?;
    let db = tensor_bias_vec(down.inner().bias())?;
    let out = packed_swiglu_ffn_f32(
        &input,
        tokens,
        gpack,
        gb.as_deref(),
        vpack,
        vb.as_deref(),
        dpack,
        db.as_deref(),
    )
    .map_err(|e| candle_core::Error::Msg(format!("swiglu ffn kernel: {e:#}")))?;
    let mut out_dims = dims;
    if let Some(last) = out_dims.last_mut() {
        *last = dpack.out_dim;
    }
    Tensor::from_vec(out, out_dims, xs.device())
}

/// Packed-inference LayerNorm (CUDA kernel when live). Training stays on Candle.
pub fn layer_norm_module(xs: &Tensor, ln: &candle_nn::LayerNorm) -> candle_core::Result<Tensor> {
    const EPS: f32 = 1e-5;
    #[cfg(feature = "cuda")]
    {
        if let Some(bank) = crate::custom_kernels::global_bank() {
            if bank.has("layer_norm") {
                let dims = xs.dims().to_vec();
                let cols = *dims.last().unwrap_or(&0);
                if cols > 0 && xs.elem_count() % cols == 0 {
                    let rows = xs.elem_count() / cols;
                    let input = xs.flatten_all()?.to_vec1::<f32>()?;
                    let gamma = ln.weight().flatten_all()?.to_vec1::<f32>()?;
                    let beta = match ln.bias() {
                        Some(b) => b.flatten_all()?.to_vec1::<f32>()?,
                        None => vec![0f32; cols],
                    };
                    if let Ok(out) = bank.layer_norm_host(&input, &gamma, &beta, rows, cols, EPS) {
                        return Tensor::from_vec(out, dims, xs.device());
                    }
                }
            }
        }
    }
    ln.forward(xs)
}

pub fn packed_mha(
    xs: &Tensor,
    q: &MaybeTernaryLinear,
    k: &MaybeTernaryLinear,
    v: &MaybeTernaryLinear,
    o: &MaybeTernaryLinear,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let (b, s, _d) = xs.dims3()?;
    let qpack = q.packed().ok_or_else(|| candle_core::Error::Msg("q not packed".into()))?;
    let kpack = k.packed().ok_or_else(|| candle_core::Error::Msg("k not packed".into()))?;
    let vpack = v.packed().ok_or_else(|| candle_core::Error::Msg("v not packed".into()))?;
    let opack = o.packed().ok_or_else(|| candle_core::Error::Msg("o not packed".into()))?;
    let input = xs.flatten_all()?.to_vec1::<f32>()?;
    let qb = tensor_bias_vec(q.inner().bias())?;
    let kb = tensor_bias_vec(k.inner().bias())?;
    let vb = tensor_bias_vec(v.inner().bias())?;
    let ob = tensor_bias_vec(o.inner().bias())?;
    let half = head_dim / 2;
    let cos = rope_cos.narrow(0, 0, s)?.flatten_all()?.to_vec1::<f32>()?;
    let sin = rope_sin.narrow(0, 0, s)?.flatten_all()?.to_vec1::<f32>()?;
    if cos.len() < s * half || sin.len() < s * half {
        candle_core::bail!("rope cache too short for seq {s}");
    }
    let mask_v = match mask {
        Some(m) => {
            let flat = m.flatten_all()?.to_vec1::<f32>()?;
            if flat.len() == b * s * s {
                Some(flat)
            } else if flat.len() == s * s {
                let mut expanded = Vec::with_capacity(b * s * s);
                for _ in 0..b {
                    expanded.extend_from_slice(&flat);
                }
                Some(expanded)
            } else {
                candle_core::bail!("attn mask len {} != B*S*S {}", flat.len(), b * s * s)
            }
        }
        None => None,
    };
    let out = packed_mha_f32(
        &input,
        b,
        s,
        qpack,
        qb.as_deref(),
        kpack,
        kb.as_deref(),
        vpack,
        vb.as_deref(),
        opack,
        ob.as_deref(),
        n_heads,
        n_kv,
        head_dim,
        &cos,
        &sin,
        mask_v.as_deref(),
    )
    .map_err(|e| candle_core::Error::Msg(format!("packed mha: {e:#}")))?;
    Tensor::from_vec(out, (b, s, opack.out_dim), xs.device())
}

pub fn packed_mha_f32(
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
) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if let Some(bank) = crate::custom_kernels::global_bank() {
            if bank.has("fused_attention") {
                if let Ok(out) = bank.packed_mha_host(
                    xs, b, s, q, q_bias, k, k_bias, v, v_bias, o, o_bias,
                    n_heads, n_kv, head_dim, rope_cos, rope_sin, mask,
                ) {
                    return Ok(out);
                }
            }
        }
    }
    Ok(kernel_ref::packed_mha(
        xs, b, s, q, q_bias, k, k_bias, v, v_bias, o, o_bias,
        n_heads, n_kv, head_dim, rope_cos, rope_sin, mask,
    ))
}

/// Host-owned LN/bias plus pack refs for one transformer block.
pub struct PackedLayerBufs<'a> {
    pub ln_attn_g: Vec<f32>,
    pub ln_attn_b: Vec<f32>,
    pub ln_ffn_g: Vec<f32>,
    pub ln_ffn_b: Vec<f32>,
    pub q: &'a TernaryPacked,
    pub q_bias: Option<Vec<f32>>,
    pub k: &'a TernaryPacked,
    pub k_bias: Option<Vec<f32>>,
    pub v: &'a TernaryPacked,
    pub v_bias: Option<Vec<f32>>,
    pub o: &'a TernaryPacked,
    pub o_bias: Option<Vec<f32>>,
    pub gate: &'a TernaryPacked,
    pub gate_bias: Option<Vec<f32>>,
    pub value: &'a TernaryPacked,
    pub value_bias: Option<Vec<f32>>,
    pub down: &'a TernaryPacked,
    pub down_bias: Option<Vec<f32>>,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

/// Packed transformer stack: activations stay on device between layers.
pub fn packed_transformer_stack(
    xs: &Tensor,
    layers: &[PackedLayerBufs<'_>],
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let (b, s, d) = xs.dims3()?;
    if layers.is_empty() {
        candle_core::bail!("packed transformer stack is empty");
    }
    let input = xs.flatten_all()?.to_vec1::<f32>()?;
    let half = layers[0].head_dim / 2;
    let cos = rope_cos.narrow(0, 0, s)?.flatten_all()?.to_vec1::<f32>()?;
    let sin = rope_sin.narrow(0, 0, s)?.flatten_all()?.to_vec1::<f32>()?;
    if cos.len() < s * half || sin.len() < s * half {
        candle_core::bail!("rope cache too short for seq {s}");
    }
    let mask_v = match mask {
        Some(m) => {
            let flat = m.flatten_all()?.to_vec1::<f32>()?;
            if flat.len() == b * s * s {
                Some(flat)
            } else if flat.len() == s * s {
                let mut expanded = Vec::with_capacity(b * s * s);
                for _ in 0..b {
                    expanded.extend_from_slice(&flat);
                }
                Some(expanded)
            } else {
                candle_core::bail!("attn mask len {} != B*S*S {}", flat.len(), b * s * s)
            }
        }
        None => None,
    };

    #[cfg(feature = "cuda")]
    {
        if let Some(bank) = crate::custom_kernels::global_bank() {
            if bank.has("fused_attention") && bank.has("layer_norm") && bank.has("ewise_add_inplace") {
                let cuda_layers: Vec<crate::custom_kernels::PackedTransformerLayer<'_>> = layers
                    .iter()
                    .map(|l| crate::custom_kernels::PackedTransformerLayer {
                        ln_attn_gamma: &l.ln_attn_g,
                        ln_attn_beta: &l.ln_attn_b,
                        ln_ffn_gamma: &l.ln_ffn_g,
                        ln_ffn_beta: &l.ln_ffn_b,
                        q: l.q,
                        q_bias: l.q_bias.as_deref(),
                        k: l.k,
                        k_bias: l.k_bias.as_deref(),
                        v: l.v,
                        v_bias: l.v_bias.as_deref(),
                        o: l.o,
                        o_bias: l.o_bias.as_deref(),
                        gate: l.gate,
                        gate_bias: l.gate_bias.as_deref(),
                        value: l.value,
                        value_bias: l.value_bias.as_deref(),
                        down: l.down,
                        down_bias: l.down_bias.as_deref(),
                        n_heads: l.n_heads,
                        n_kv: l.n_kv,
                        head_dim: l.head_dim,
                    })
                    .collect();
                if let Ok(out) = bank.packed_transformer_stack_host(
                    &input,
                    b,
                    s,
                    d,
                    &cuda_layers,
                    &cos,
                    &sin,
                    mask_v.as_deref(),
                    1e-5,
                ) {
                    return Tensor::from_vec(out, (b, s, d), xs.device());
                }
            }
        }
    }
    let _ = (input, cos, sin, mask_v, b, s, d);
    candle_core::bail!("packed transformer stack unavailable")
}

pub fn linear_maybe_ternary(
    xs: &Tensor,
    lin: &Linear,
    packed: Option<&TernaryPacked>,
) -> candle_core::Result<Tensor> {
    match packed {
        Some(pack) => packed_linear(xs, pack, lin.bias()),
        None => lin.forward(xs),
    }
}

/// Candle `Linear` plus optional packed weights for the kernel path.
pub struct MaybeTernaryLinear {
    inner: Linear,
    packed: Option<TernaryPacked>,
}

impl MaybeTernaryLinear {
    pub fn new(inner: Linear) -> Self {
        Self { inner, packed: None }
    }

    pub fn inner(&self) -> &Linear {
        &self.inner
    }

    pub fn packed(&self) -> Option<&TernaryPacked> {
        self.packed.as_ref()
    }

    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }

    pub fn pack_from_weight(&mut self) -> Result<()> {
        let w = self.inner.weight();
        let (_, in_dim) = w.dims2()?;
        if in_dim % 4 != 0 {
            self.packed = None;
            return Ok(());
        }
        self.packed = Some(tensor_to_ternary(w)?);
        Ok(())
    }

    pub fn set_packed(&mut self, pack: TernaryPacked) {
        self.packed = Some(pack);
    }

    pub fn apply_named(&mut self, prefix: &str, packs: &std::collections::BTreeMap<String, TernaryPacked>) {
        let key = format!("{prefix}.weight");
        if let Some(p) = packs.get(&key) {
            self.packed = Some(p.clone());
        }
    }

    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        linear_maybe_ternary(xs, &self.inner, self.packed.as_ref())
    }
}
