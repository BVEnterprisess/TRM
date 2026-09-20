#![cfg(feature = "cuda")]
//! Custom CUDA kernels loader via cudarc.
//!
//! Training still runs through Candle autograd. These kernels are the
//! inference / quantized-GEMM path. If PTX was not compiled (no nvcc),
//! `try_new` fails and callers fall back to `kernel_ref`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use serde::Serialize;

use crate::quantize::TernaryPacked;

/// KernelBank HtoD/DtoH counters. Reset around a timed region; ignore warmup.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuCopyStats {
    pub htod_calls: u64,
    pub dtoh_calls: u64,
    pub htod_bytes: u64,
    pub dtoh_bytes: u64,
    pub htod_mb: f64,
    pub dtoh_mb: f64,
}

struct CopyCounters {
    htod_calls: AtomicU64,
    dtoh_calls: AtomicU64,
    htod_bytes: AtomicU64,
    dtoh_bytes: AtomicU64,
}

impl CopyCounters {
    fn new() -> Self {
        Self {
            htod_calls: AtomicU64::new(0),
            dtoh_calls: AtomicU64::new(0),
            htod_bytes: AtomicU64::new(0),
            dtoh_bytes: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.htod_calls.store(0, Ordering::Relaxed);
        self.dtoh_calls.store(0, Ordering::Relaxed);
        self.htod_bytes.store(0, Ordering::Relaxed);
        self.dtoh_bytes.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> GpuCopyStats {
        let htod_bytes = self.htod_bytes.load(Ordering::Relaxed);
        let dtoh_bytes = self.dtoh_bytes.load(Ordering::Relaxed);
        GpuCopyStats {
            htod_calls: self.htod_calls.load(Ordering::Relaxed),
            dtoh_calls: self.dtoh_calls.load(Ordering::Relaxed),
            htod_bytes,
            dtoh_bytes,
            htod_mb: htod_bytes as f64 / (1024.0 * 1024.0),
            dtoh_mb: dtoh_bytes as f64 / (1024.0 * 1024.0),
        }
    }
}

/// One packed transformer block. Packs must outlive the stack call.
pub struct PackedTransformerLayer<'a> {
    pub ln_attn_gamma: &'a [f32],
    pub ln_attn_beta: &'a [f32],
    pub ln_ffn_gamma: &'a [f32],
    pub ln_ffn_beta: &'a [f32],
    pub q: &'a TernaryPacked,
    pub q_bias: Option<&'a [f32]>,
    pub k: &'a TernaryPacked,
    pub k_bias: Option<&'a [f32]>,
    pub v: &'a TernaryPacked,
    pub v_bias: Option<&'a [f32]>,
    pub o: &'a TernaryPacked,
    pub o_bias: Option<&'a [f32]>,
    pub gate: &'a TernaryPacked,
    pub gate_bias: Option<&'a [f32]>,
    pub value: &'a TernaryPacked,
    pub value_bias: Option<&'a [f32]>,
    pub down: &'a TernaryPacked,
    pub down_bias: Option<&'a [f32]>,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

struct DevicePack {
    weights: CudaSlice<u8>,
    alphas: CudaSlice<f32>,
    bias: Option<CudaSlice<f32>>,
    in_dim: usize,
    out_dim: usize,
}

pub struct KernelBank {
    pub dev: Arc<CudaDevice>,
    funcs: HashMap<String, CudaFunction>,
    packs: Mutex<HashMap<usize, Arc<DevicePack>>>,
    copy_stats: CopyCounters,
}

static BANK: OnceLock<Option<KernelBank>> = OnceLock::new();

/// Process-wide bank. `None` if the driver or PTX modules are unavailable.
pub fn global_bank() -> Option<&'static KernelBank> {
    BANK.get_or_init(|| KernelBank::try_new(0).ok()).as_ref()
}

/// Existing bank only — does not create a CUDA context just to sample VRAM.
pub fn global_bank_if_ready() -> Option<&'static KernelBank> {
    BANK.get().and_then(|opt| opt.as_ref())
}

fn gemm_launch(tokens: usize, out_dim: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (tokens as u32, ((out_dim as u32 + 255) / 256).max(1), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn elem_launch(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn is_stub_ptx(src: &str) -> bool {
    let trimmed = src.trim();
    trimmed.is_empty() || trimmed.contains("TRM_PTX_STUB")
}

fn resolve_ptx<'a>(compiled: &'a str, vendor: &'a str) -> &'a str {
    if is_stub_ptx(compiled) {
        vendor
    } else {
        compiled
    }
}

impl KernelBank {
    pub fn try_new(ordinal: usize) -> Result<Self> {
        let dev = CudaDevice::new(ordinal)
            .with_context(|| format!("failed to init CUDA device {ordinal}"))?;
        let mut this = Self {
            dev: dev.clone(),
            funcs: HashMap::new(),
            packs: Mutex::new(HashMap::new()),
            copy_stats: CopyCounters::new(),
        };

        let omega = resolve_ptx(
            include_str!(concat!(env!("OUT_DIR"), "/omega.ptx")),
            include_str!("kernels/fallback/omega.ptx"),
        );
        this.load("omega", omega, &["ternary_matmul", "ternary_matmul_stack"])
            .context("loading omega ternary kernels")?;

        let activations = resolve_ptx(
            include_str!(concat!(env!("OUT_DIR"), "/activations.ptx")),
            include_str!("kernels/fallback/activations.ptx"),
        );
        let act_all = [
            "swiglu_fused", "relu_inplace", "gelu_inplace", "silu_inplace",
            "ewise_add", "ewise_add_inplace", "ewise_mul", "ewise_add_scaled",
            "bias_add", "copy_buffer", "fill_const",
            "layer_norm", "layer_norm_inplace", "softmax_row",
            "cross_entropy_loss", "check_nan_inf",
        ];
        if this.load("activations", activations, &act_all).is_err() {
            let _ = this.load(
                "activations",
                activations,
                &["swiglu_fused", "bias_add", "check_nan_inf", "layer_norm"],
            );
        }

        let _ = this.load(
            "attention",
            include_str!(concat!(env!("OUT_DIR"), "/attention.ptx")),
            &["precompute_rope", "apply_rope",
              "transpose_bshd_to_bhsd", "transpose_bhsd_to_bsd",
              "fused_attention", "dense_matmul_bias"],
        );
        let _ = this.load(
            "mixer",
            include_str!(concat!(env!("OUT_DIR"), "/mixer.ptx")),
            &["transpose_bsd_to_bds", "transpose_bds_to_bsd",
              "swiglu_up_fused", "dense_down"],
        );
        let _ = this.load(
            "recurse",
            include_str!(concat!(env!("OUT_DIR"), "/recurse.ptx")),
            &["halt_confidence", "update_halt_mask", "deep_supervision_l2",
              "slice_copy", "write_subseq", "argmax_per_pos",
              "add_segment_embed", "zero_agent_state", "ema_update"],
        );

        if !this.funcs.contains_key("ternary_matmul_stack") {
            anyhow::bail!("KernelBank missing ternary_matmul_stack");
        }
        Ok(this)
    }

    pub fn new(ordinal: usize) -> Result<Self> {
        Self::try_new(ordinal)
    }

    fn load(&mut self, module: &str, ptx_src: &str, funcs: &[&'static str]) -> Result<()> {
        if is_stub_ptx(ptx_src) {
            anyhow::bail!("PTX stub for {module}");
        }
        self.dev.load_ptx(Ptx::from_src(ptx_src.replace('\r', "")), module, funcs)
            .with_context(|| format!("loading module {module}"))?;
        for &f in funcs {
            if let Some(func) = self.dev.get_func(module, f) {
                self.funcs.insert(f.to_string(), func);
            }
        }
        Ok(())
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        self.funcs.get(name).cloned().ok_or_else(|| anyhow!("kernel {name} not loaded"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.funcs.contains_key(name)
    }

    pub fn loaded_kernels(&self) -> Vec<String> {
        let mut names: Vec<String> = self.funcs.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn htod<T: cudarc::driver::DeviceRepr + Unpin + Clone>(&self, data: &[T]) -> Result<CudaSlice<T>> {
        let bytes = (std::mem::size_of::<T>() * data.len()) as u64;
        self.copy_stats.htod_calls.fetch_add(1, Ordering::Relaxed);
        self.copy_stats.htod_bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(self.dev.htod_sync_copy(data)?)
    }
    pub fn dtoh<T: cudarc::driver::DeviceRepr + Default + Clone + Unpin>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>> {
        let bytes = (std::mem::size_of::<T>() * buf.len()) as u64;
        self.copy_stats.dtoh_calls.fetch_add(1, Ordering::Relaxed);
        self.copy_stats.dtoh_bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(self.dev.dtoh_sync_copy(buf)?)
    }
    pub fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Clone>(
        &self,
        n: usize,
    ) -> Result<CudaSlice<T>> {
        Ok(self.dev.alloc_zeros::<T>(n)?)
    }
    pub fn sync(&self) -> Result<()> { Ok(self.dev.synchronize()?) }

    pub fn reset_copy_stats(&self) {
        self.copy_stats.reset();
    }

    pub fn snapshot_copy_stats(&self) -> GpuCopyStats {
        self.copy_stats.snapshot()
    }

    /// CUDA-visible `(free, total)` bytes for the current device.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.sync()?;
        cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow!("cuMemGetInfo: {e:?}"))
    }

    pub fn ternary_matmul(
        &self,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<u8>,
        alphas: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        apply_relu: bool,
    ) -> Result<()> {
        let cfg = gemm_launch(batch, out_dim);
        let f = self.func("ternary_matmul")?;
        unsafe {
            f.launch(cfg, (
                input, weights, alphas, output,
                in_dim as i32, out_dim as i32,
                if apply_relu { 1i32 } else { 0i32 },
            ))?;
        }
        Ok(())
    }

    pub fn ternary_matmul_stack(
        &self,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<u8>,
        alphas: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        total_tokens: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        let cfg = gemm_launch(total_tokens, out_dim);
        let f = self.func("ternary_matmul_stack")?;
        unsafe {
            f.launch(cfg, (
                input, weights, alphas, output,
                total_tokens as i32, in_dim as i32, out_dim as i32,
            ))?;
        }
        Ok(())
    }

    pub fn ternary_matmul_host(
        &self,
        input: &[f32],
        pack: &TernaryPacked,
        batch: usize,
        apply_relu: bool,
    ) -> Result<Vec<f32>> {
        if input.len() != batch * pack.in_dim {
            anyhow::bail!(
                "ternary_matmul input len {} != batch {batch} * in_dim {}",
                input.len(),
                pack.in_dim
            );
        }
        let inp = self.htod(input)?;
        let w = self.htod(&pack.packed)?;
        let a = self.htod(&pack.alphas)?;
        let mut out = self.zeros::<f32>(batch * pack.out_dim)?;
        self.ternary_matmul(&inp, &w, &a, &mut out, batch, pack.in_dim, pack.out_dim, apply_relu)?;
        self.sync()?;
        self.dtoh(&out)
    }

    fn pack_key(pack: &TernaryPacked) -> usize {
        pack.packed.as_ptr() as usize
    }

    fn cached_pack(&self, pack: &TernaryPacked, bias: Option<&[f32]>) -> Result<()> {
        let key = Self::pack_key(pack);
        let mut cache = self.packs.lock().map_err(|e| anyhow!("pack cache: {e}"))?;
        if let Some(existing) = cache.get_mut(&key) {
            if existing.bias.is_none() {
                if let Some(b) = bias {
                    if let Some(dp) = Arc::get_mut(existing) {
                        dp.bias = Some(self.htod(b)?);
                    }
                }
            }
            return Ok(());
        }
        cache.insert(
            key,
            Arc::new(DevicePack {
                weights: self.htod(&pack.packed)?,
                alphas: self.htod(&pack.alphas)?,
                bias: match bias {
                    Some(b) => Some(self.htod(b)?),
                    None => None,
                },
                in_dim: pack.in_dim,
                out_dim: pack.out_dim,
            }),
        );
        Ok(())
    }

    fn device_pack(&self, pack: &TernaryPacked) -> Result<Arc<DevicePack>> {
        let cache = self.packs.lock().map_err(|e| anyhow!("pack cache: {e}"))?;
        cache
            .get(&Self::pack_key(pack))
            .cloned()
            .ok_or_else(|| anyhow!("device pack missing; call cached_pack first"))
    }

    fn bias_add_device(
        &self,
        data: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, ((cols as u32 + 255) / 256).max(1), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.func("bias_add")?;
        unsafe {
            f.launch(cfg, (data, bias, rows as i32, cols as i32))?;
        }
        Ok(())
    }

    fn linear_device(
        &self,
        input: &CudaSlice<f32>,
        pack: &TernaryPacked,
        tokens: usize,
    ) -> Result<CudaSlice<f32>> {
        let dp = self.device_pack(pack)?;
        let mut out = self.zeros::<f32>(tokens * dp.out_dim)?;
        self.ternary_matmul_stack(input, &dp.weights, &dp.alphas, &mut out, tokens, dp.in_dim, dp.out_dim)?;
        if let Some(bias) = &dp.bias {
            self.bias_add_device(&mut out, bias, tokens, dp.out_dim)?;
        }
        Ok(out)
    }

    pub fn ternary_matmul_stack_host(
        &self,
        input: &[f32],
        pack: &TernaryPacked,
        tokens: usize,
    ) -> Result<Vec<f32>> {
        self.packed_linear_host(input, pack, None, tokens)
    }

    /// GEMM + optional bias with weights cached on device after the first call.
    pub fn packed_linear_host(
        &self,
        input: &[f32],
        pack: &TernaryPacked,
        bias: Option<&[f32]>,
        tokens: usize,
    ) -> Result<Vec<f32>> {
        if input.len() != tokens * pack.in_dim {
            anyhow::bail!(
                "packed_linear input len {} != tokens {tokens} * in_dim {}",
                input.len(),
                pack.in_dim
            );
        }
        self.cached_pack(pack, bias)?;
        let inp = self.htod(input)?;
        let out = self.linear_device(&inp, pack, tokens)?;
        self.sync()?;
        self.dtoh(&out)
    }

    fn ewise_add_inplace_device(&self, a: &mut CudaSlice<f32>, b: &CudaSlice<f32>, n: usize) -> Result<()> {
        let f = self.func("ewise_add_inplace")?;
        unsafe {
            f.launch(elem_launch(n), (a, b, n as i32))?;
        }
        Ok(())
    }

    fn layer_norm_device(
        &self,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>> {
        let mut out = self.zeros::<f32>(rows * cols)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.func("layer_norm")?;
        unsafe {
            f.launch(cfg, (input, gamma, beta, &mut out, rows as i32, cols as i32, eps))?;
        }
        Ok(out)
    }

    fn packed_swiglu_ffn_device(
        &self,
        xs: &CudaSlice<f32>,
        tokens: usize,
        gate: &TernaryPacked,
        value: &TernaryPacked,
        down: &TernaryPacked,
    ) -> Result<CudaSlice<f32>> {
        let g = self.linear_device(xs, gate, tokens)?;
        let v = self.linear_device(xs, value, tokens)?;
        let hidden_n = tokens * gate.out_dim;
        let mut hidden = self.zeros::<f32>(hidden_n)?;
        let f = self.func("swiglu_fused")?;
        unsafe {
            f.launch(elem_launch(hidden_n), (&g, &v, &mut hidden, hidden_n as i32))?;
        }
        self.linear_device(&hidden, down, tokens)
    }

    /// gate/value GEMM, fused silu*up, down GEMM — one HtoD / DtoH pair.
    pub fn packed_swiglu_ffn_host(
        &self,
        xs: &[f32],
        tokens: usize,
        gate: &TernaryPacked,
        gate_bias: Option<&[f32]>,
        value: &TernaryPacked,
        value_bias: Option<&[f32]>,
        down: &TernaryPacked,
        down_bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        self.cached_pack(gate, gate_bias)?;
        self.cached_pack(value, value_bias)?;
        self.cached_pack(down, down_bias)?;
        let inp = self.htod(xs)?;
        let out = self.packed_swiglu_ffn_device(&inp, tokens, gate, value, down)?;
        self.sync()?;
        self.dtoh(&out)
    }

    fn transpose_bshd_to_bhsd(
        &self,
        src: &CudaSlice<f32>,
        b: usize,
        s: usize,
        h: usize,
        d: usize,
    ) -> Result<CudaSlice<f32>> {
        let n = b * s * h * d;
        let mut dst = self.zeros::<f32>(n)?;
        let f = self.func("transpose_bshd_to_bhsd")?;
        unsafe {
            f.launch(elem_launch(n), (src, &mut dst, b as i32, s as i32, h as i32, d as i32))?;
        }
        Ok(dst)
    }

    fn transpose_bhsd_to_bsd(
        &self,
        src: &CudaSlice<f32>,
        b: usize,
        h: usize,
        s: usize,
        d: usize,
    ) -> Result<CudaSlice<f32>> {
        let n = b * h * s * d;
        let mut dst = self.zeros::<f32>(n)?;
        let f = self.func("transpose_bhsd_to_bsd")?;
        unsafe {
            f.launch(elem_launch(n), (src, &mut dst, b as i32, h as i32, s as i32, d as i32))?;
        }
        Ok(dst)
    }

    fn apply_rope_device(
        &self,
        x: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        batch: usize,
        heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<()> {
        let half = (head_dim / 2).max(1);
        let cfg = LaunchConfig {
            grid_dim: ((batch * heads * seq) as u32, 1, 1),
            block_dim: (half as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.func("apply_rope")?;
        unsafe {
            f.launch(cfg, (x, cos, sin, batch as i32, heads as i32, seq as i32, head_dim as i32))?;
        }
        Ok(())
    }

    fn packed_mha_device(
        &self,
        inp: &CudaSlice<f32>,
        b: usize,
        s: usize,
        q: &TernaryPacked,
        k: &TernaryPacked,
        v: &TernaryPacked,
        o: &TernaryPacked,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        rope_cos: &CudaSlice<f32>,
        rope_sin: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        let tokens = b * s;
        let q_bsd = self.linear_device(inp, q, tokens)?;
        let k_bsd = self.linear_device(inp, k, tokens)?;
        let v_bsd = self.linear_device(inp, v, tokens)?;
        let mut qh = self.transpose_bshd_to_bhsd(&q_bsd, b, s, n_heads, head_dim)?;
        let mut kh = self.transpose_bshd_to_bhsd(&k_bsd, b, s, n_kv, head_dim)?;
        let vh = self.transpose_bshd_to_bhsd(&v_bsd, b, s, n_kv, head_dim)?;
        self.apply_rope_device(&mut qh, rope_cos, rope_sin, b, n_heads, s, head_dim)?;
        self.apply_rope_device(&mut kh, rope_cos, rope_sin, b, n_kv, s, head_dim)?;
        let mut ctx = self.zeros::<f32>(b * n_heads * s * head_dim)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let queries = b * n_heads * s;
        let cfg = LaunchConfig {
            grid_dim: (queries as u32, 1, 1),
            block_dim: (head_dim.max(1) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.func("fused_attention")?;
        unsafe {
            f.launch(cfg, (
                &qh, &kh, &vh, mask, &mut ctx,
                b as i32, n_heads as i32, n_kv as i32, s as i32, head_dim as i32, scale,
            ))?;
        }
        let ctx_bsd = self.transpose_bhsd_to_bsd(&ctx, b, n_heads, s, head_dim)?;
        self.linear_device(&ctx_bsd, o, tokens)
    }

    /// Packed QKV + RoPE + fused attention + O, weights resident after first call.
    pub fn packed_mha_host(
        &self,
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
        if !self.has("fused_attention") || !self.has("apply_rope") {
            anyhow::bail!("attention kernels not loaded");
        }
        if n_kv == 0 || n_heads % n_kv != 0 {
            anyhow::bail!("GQA requires n_heads % n_kv == 0");
        }
        self.cached_pack(q, q_bias)?;
        self.cached_pack(k, k_bias)?;
        self.cached_pack(v, v_bias)?;
        self.cached_pack(o, o_bias)?;
        let inp = self.htod(xs)?;
        let cos = self.htod(rope_cos)?;
        let sin = self.htod(rope_sin)?;
        let mask_host = match mask {
            Some(m) => m.to_vec(),
            None => vec![0f32; b * s * s],
        };
        let mask_d = self.htod(&mask_host)?;
        let out = self.packed_mha_device(
            &inp, b, s, q, k, v, o, n_heads, n_kv, head_dim, &cos, &sin, &mask_d,
        )?;
        self.sync()?;
        self.dtoh(&out)
    }

    /// Full transformer stack: one activation HtoD, kernels stay on device, one DtoH.
    pub fn packed_transformer_stack_host(
        &self,
        xs: &[f32],
        batch: usize,
        seq: usize,
        dim: usize,
        layers: &[PackedTransformerLayer<'_>],
        rope_cos: &[f32],
        rope_sin: &[f32],
        mask: Option<&[f32]>,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if layers.is_empty() {
            anyhow::bail!("packed transformer stack needs at least one layer");
        }
        if !self.has("fused_attention") || !self.has("apply_rope") || !self.has("layer_norm") {
            anyhow::bail!("transformer stack kernels not loaded");
        }
        if !self.has("ewise_add_inplace") {
            anyhow::bail!("ewise_add_inplace not loaded");
        }
        let tokens = batch * seq;
        if xs.len() != tokens * dim {
            anyhow::bail!("stack input len {} != B*S*D {}", xs.len(), tokens * dim);
        }
        for layer in layers {
            if layer.n_kv == 0 || layer.n_heads % layer.n_kv != 0 {
                anyhow::bail!("GQA requires n_heads % n_kv == 0");
            }
            self.cached_pack(layer.q, layer.q_bias)?;
            self.cached_pack(layer.k, layer.k_bias)?;
            self.cached_pack(layer.v, layer.v_bias)?;
            self.cached_pack(layer.o, layer.o_bias)?;
            self.cached_pack(layer.gate, layer.gate_bias)?;
            self.cached_pack(layer.value, layer.value_bias)?;
            self.cached_pack(layer.down, layer.down_bias)?;
        }
        let mut x = self.htod(xs)?;
        let cos = self.htod(rope_cos)?;
        let sin = self.htod(rope_sin)?;
        let mask_host = match mask {
            Some(m) => m.to_vec(),
            None => vec![0f32; batch * seq * seq],
        };
        let mask_d = self.htod(&mask_host)?;
        let n = tokens * dim;
        for layer in layers {
            let g_attn = self.htod(layer.ln_attn_gamma)?;
            let b_attn = self.htod(layer.ln_attn_beta)?;
            let ln_attn = self.layer_norm_device(&x, &g_attn, &b_attn, tokens, dim, eps)?;
            let attn = self.packed_mha_device(
                &ln_attn, batch, seq, layer.q, layer.k, layer.v, layer.o,
                layer.n_heads, layer.n_kv, layer.head_dim, &cos, &sin, &mask_d,
            )?;
            self.ewise_add_inplace_device(&mut x, &attn, n)?;
            let g_ffn = self.htod(layer.ln_ffn_gamma)?;
            let b_ffn = self.htod(layer.ln_ffn_beta)?;
            let ln_ffn = self.layer_norm_device(&x, &g_ffn, &b_ffn, tokens, dim, eps)?;
            let ffn = self.packed_swiglu_ffn_device(&ln_ffn, tokens, layer.gate, layer.value, layer.down)?;
            self.ewise_add_inplace_device(&mut x, &ffn, n)?;
        }
        self.sync()?;
        self.dtoh(&x)
    }

    pub fn swiglu_fused_host(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            anyhow::bail!("swiglu gate/up length mismatch");
        }
        let n = gate.len();
        let g = self.htod(gate)?;
        let u = self.htod(up)?;
        let mut out = self.zeros::<f32>(n)?;
        let f = self.func("swiglu_fused")?;
        unsafe {
            f.launch(elem_launch(n), (&g, &u, &mut out, n as i32))?;
        }
        self.sync()?;
        self.dtoh(&out)
    }

    pub fn layer_norm_host(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if input.len() != rows * cols {
            anyhow::bail!("layer_norm input size");
        }
        let inp = self.htod(input)?;
        let g = self.htod(gamma)?;
        let b = self.htod(beta)?;
        let mut out = self.zeros::<f32>(rows * cols)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.func("layer_norm")?;
        unsafe {
            f.launch(cfg, (&inp, &g, &b, &mut out, rows as i32, cols as i32, eps))?;
        }
        self.sync()?;
        self.dtoh(&out)
    }

    pub fn dense_matmul_bias_host(
        &self,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let bias_host = match bias {
            Some(bias) => bias.to_vec(),
            None => vec![0f32; n],
        };
        let a_d = self.htod(a)?;
        let b_d = self.htod(b)?;
        let bias_d = self.htod(&bias_host)?;
        let mut c = self.zeros::<f32>(m * n)?;
        let bx = 16u32;
        let by = 16u32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + bx - 1) / bx, (m as u32 + by - 1) / by, 1),
            block_dim: (bx, by, 1),
            shared_mem_bytes: 0,
        };
        let f = self.func("dense_matmul_bias")?;
        unsafe {
            f.launch(cfg, (&a_d, &b_d, &bias_d, &mut c, m as i32, k as i32, n as i32))?;
        }
        self.sync()?;
        self.dtoh(&c)
    }

    pub fn check_nan_inf(&self, x: &CudaSlice<f32>, n: usize) -> Result<(bool, bool)> {
        let flags = vec![0i32; 2];
        let flags_dev = self.dev.htod_sync_copy(flags.as_slice())?;
        let f = self.func("check_nan_inf")?;
        unsafe {
            f.launch(elem_launch(n), (x, &flags_dev, n as i32))?;
        }
        self.dev.synchronize()?;
        let result = self.dev.dtoh_sync_copy(&flags_dev)?;
        Ok((result[0] != 0, result[1] != 0))
    }

    pub fn bias_add_host(&self, data: &mut [f32], bias: &[f32], rows: usize, cols: usize) -> Result<()> {
        if data.len() != rows * cols || bias.len() != cols {
            anyhow::bail!("bias_add shape");
        }
        let mut buf = self.htod(data)?;
        let b = self.htod(bias)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, ((cols as u32 + 255) / 256).max(1), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.func("bias_add")?;
        unsafe {
            f.launch(cfg, (&mut buf, &b, rows as i32, cols as i32))?;
        }
        self.sync()?;
        let out = self.dtoh(&buf)?;
        data.copy_from_slice(&out);
        Ok(())
    }
}
