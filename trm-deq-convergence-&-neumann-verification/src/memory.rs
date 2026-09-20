//! # Memory and Agent State Management
//!
//! GPU memory is tight on a GTX 1660 (6GB). This module provides:
//! - VRAM budget estimation
//! - Batched persistent state storage for multi-agent inference
//! - Selective agent reset / fork / serialization helpers

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub total_vram_mb: usize,
    pub reserve_for_system_mb: usize,
    pub usable_vram_mb: usize,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        let total = 6 * 1024usize;
        let reserve = 1024usize;
        Self {
            total_vram_mb: total,
            reserve_for_system_mb: reserve,
            usable_vram_mb: total.saturating_sub(reserve),
        }
    }
}

impl MemoryBudget {
    pub fn estimate_activation_mb(
        &self,
        batch: usize,
        seq: usize,
        dim: usize,
        buffers: usize,
    ) -> f64 {
        (batch * seq * dim * 4 * buffers) as f64 / 1024.0 / 1024.0
    }

    pub fn recommend_batch_size(&self, seq: usize, dim: usize, buffers: usize) -> usize {
        let per_batch_mb = self.estimate_activation_mb(1, seq, dim, buffers).max(0.001);
        ((self.usable_vram_mb as f64 / per_batch_mb) as usize).max(1)
    }
}

#[derive(Clone)]
pub struct AgentBatchState {
    pub y: Tensor,
    pub z: Tensor,
    pub halted: Vec<bool>,
    pub confidence: Vec<f32>,
    pub y_seq: usize,
    pub z_seq: usize,
    pub dim: usize,
}

impl AgentBatchState {
    pub fn new(device: &Device, batch: usize, y_seq: usize, z_seq: usize, dim: usize) -> Result<Self> {
        let y = Tensor::zeros((batch, y_seq, dim), DType::F32, device)?;
        let z = Tensor::zeros((batch, z_seq, dim), DType::F32, device)?;
        Ok(Self {
            y,
            z,
            halted: vec![false; batch],
            confidence: vec![0.0; batch],
            y_seq,
            z_seq,
            dim,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.halted.len()
    }

    pub fn reset_all(&mut self, device: &Device) -> Result<()> {
        self.y = Tensor::zeros((self.batch_size(), self.y_seq, self.dim), DType::F32, device)?;
        self.z = Tensor::zeros((self.batch_size(), self.z_seq, self.dim), DType::F32, device)?;
        self.halted.fill(false);
        self.confidence.fill(0.0);
        Ok(())
    }

    pub fn reset_agent(&mut self, device: &Device, agent_idx: usize) -> Result<()> {
        let b = self.batch_size();
        if agent_idx >= b {
            anyhow::bail!("agent_idx {} out of range {}", agent_idx, b);
        }
        let mut y = self.y.flatten_all()?.to_vec1::<f32>()?;
        let mut z = self.z.flatten_all()?.to_vec1::<f32>()?;
        let y_elems = self.y_seq * self.dim;
        let z_elems = self.z_seq * self.dim;
        let yo = agent_idx * y_elems;
        let zo = agent_idx * z_elems;
        y[yo..yo + y_elems].fill(0.0);
        z[zo..zo + z_elems].fill(0.0);
        self.y = Tensor::from_vec(y, (b, self.y_seq, self.dim), device)?;
        self.z = Tensor::from_vec(z, (b, self.z_seq, self.dim), device)?;
        self.halted[agent_idx] = false;
        self.confidence[agent_idx] = 0.0;
        Ok(())
    }

    pub fn set_halt_probs(&mut self, probs: &[f32], threshold: f32) {
        for (i, &p) in probs.iter().enumerate() {
            self.confidence[i] = p;
            if p >= threshold {
                self.halted[i] = true;
            }
        }
    }

    pub fn active_indices(&self) -> Vec<usize> {
        self.halted
            .iter()
            .enumerate()
            .filter_map(|(i, &halted)| if halted { None } else { Some(i) })
            .collect()
    }

    pub fn fork(&self) -> Self {
        self.clone()
    }

    pub fn save_npz_like<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let y = self.y.flatten_all()?.to_vec1::<f32>()?;
        let z = self.z.flatten_all()?.to_vec1::<f32>()?;
        let payload = serde_json::json!({
            "shape_y": [self.batch_size(), self.y_seq, self.dim],
            "shape_z": [self.batch_size(), self.z_seq, self.dim],
            "halted": self.halted,
            "confidence": self.confidence,
            "y": y,
            "z": z,
        });
        fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramSample {
    pub name: String,
    pub total_mb: f64,
    pub used_mb: f64,
    pub free_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessVram {
    pub pid: u32,
    pub used_mb: f64,
    /// How `used_mb` was measured. WDDM does not expose this via nvidia-smi.
    #[serde(default)]
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedicated_mb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_mb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_free_mb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_total_mb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramReport {
    pub gpu: Option<VramSample>,
    pub process: Option<ProcessVram>,
    pub budget: MemoryBudgetSnapshot,
    pub inference_est_mb: f64,
    pub training_est_mb: f64,
    pub readme_inference_mb: [f64; 2],
    pub readme_training_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudgetSnapshot {
    pub total_vram_mb: usize,
    pub usable_vram_mb: usize,
}

impl From<&MemoryBudget> for MemoryBudgetSnapshot {
    fn from(b: &MemoryBudget) -> Self {
        Self {
            total_vram_mb: b.total_vram_mb,
            usable_vram_mb: b.usable_vram_mb,
        }
    }
}

fn parse_csv_row(line: &str) -> Vec<String> {
    line.split(',').map(|s| s.trim().to_string()).collect()
}

/// Query the first GPU via `nvidia-smi`. Returns `None` if the binary is missing.
pub fn sample_nvidia_smi() -> Option<VramSample> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout).lines().next()?.to_string();
    let cols = parse_csv_row(&line);
    if cols.len() < 4 {
        return None;
    }
    Some(VramSample {
        name: cols[0].clone(),
        total_mb: cols[1].parse().ok()?,
        used_mb: cols[2].parse().ok()?,
        free_mb: cols[3].parse().ok()?,
    })
}

fn bytes_to_mb(bytes: f64) -> f64 {
    bytes / 1024.0 / 1024.0
}

fn sample_cuda_mem_info() -> Option<(f64, f64)> {
    #[cfg(feature = "cuda")]
    {
        let bank = crate::custom_kernels::global_bank_if_ready()?;
        let (free, total) = bank.mem_info().ok()?;
        Some((bytes_to_mb(free as f64), bytes_to_mb(total as f64)))
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// nvidia-smi per-process bytes are N/A under WDDM. Skip non-numeric rows.
fn sample_nvidia_smi_process(pid: u32) -> Option<f64> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let cols = parse_csv_row(line);
        if cols.len() < 2 {
            continue;
        }
        if cols[0].parse::<u32>().ok() != Some(pid) {
            continue;
        }
        let raw = cols[1].replace("[N/A]", "").replace("N/A", "").trim().to_string();
        if raw.is_empty() || raw.contains("Insufficient") {
            return None;
        }
        return raw.parse().ok();
    }
    None
}

/// This process's dedicated GPU memory (Task Manager / VidMm), plus CUDA device totals.
pub fn sample_this_process_vram() -> Option<ProcessVram> {
    let pid = std::process::id();
    let cuda = sample_cuda_mem_info();
    let attach_cuda = |mut p: ProcessVram| {
        if let Some((free, total)) = cuda {
            p.cuda_free_mb = Some(free);
            p.cuda_total_mb = Some(total);
        }
        p
    };

    #[cfg(windows)]
    if let Some((dedicated, shared)) = wddm::process_gpu_bytes(pid) {
        if dedicated > 0.0 || shared > 0.0 || cuda.is_some() {
            return Some(attach_cuda(ProcessVram {
                pid,
                used_mb: bytes_to_mb(dedicated),
                source: "wddm-vidmm".into(),
                dedicated_mb: Some(bytes_to_mb(dedicated)),
                shared_mb: Some(bytes_to_mb(shared)),
                cuda_free_mb: None,
                cuda_total_mb: None,
            }));
        }
    }

    if let Some(used_mb) = sample_nvidia_smi_process(pid) {
        return Some(attach_cuda(ProcessVram {
            pid,
            used_mb,
            source: "nvidia-smi".into(),
            dedicated_mb: None,
            shared_mb: None,
            cuda_free_mb: None,
            cuda_total_mb: None,
        }));
    }

    if let Some((free, total)) = cuda {
        return Some(ProcessVram {
            pid,
            used_mb: (total - free).max(0.0),
            source: "cuda-mem-get-info".into(),
            dedicated_mb: None,
            shared_mb: None,
            cuda_free_mb: Some(free),
            cuda_total_mb: Some(total),
        });
    }
    None
}

#[cfg(windows)]
mod wddm {
    use std::time::Duration;

    type PdhStatus = i32;
    type PdhHandle = isize;

    const PDH_MORE_DATA: u32 = 0x8000_07D2;
    const PDH_FMT_DOUBLE: u32 = 0x0000_0200;

    #[link(name = "pdh")]
    extern "system" {
        fn PdhOpenQueryW(data_source: *const u16, user_data: usize, query: *mut PdhHandle) -> PdhStatus;
        fn PdhAddEnglishCounterW(
            query: PdhHandle,
            path: *const u16,
            user_data: usize,
            counter: *mut PdhHandle,
        ) -> PdhStatus;
        fn PdhCollectQueryData(query: PdhHandle) -> PdhStatus;
        fn PdhGetFormattedCounterArrayW(
            counter: PdhHandle,
            format: u32,
            buffer_size: *mut u32,
            item_count: *mut u32,
            buffer: *mut u8,
        ) -> PdhStatus;
        fn PdhCloseQuery(query: PdhHandle) -> PdhStatus;
    }

    #[repr(C)]
    struct PdhFmtValue {
        c_status: u32,
        _pad: u32,
        value: f64,
    }

    #[repr(C)]
    struct PdhFmtItem {
        name: *mut u16,
        value: PdhFmtValue,
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn ok(status: PdhStatus) -> bool {
        status == 0
    }

    fn more_data(status: PdhStatus) -> bool {
        status as u32 == PDH_MORE_DATA
    }

    fn sum_pid_bytes(counter: PdhHandle, pid: u32) -> Option<f64> {
        let mut size = 0u32;
        let mut count = 0u32;
        let st = unsafe {
            PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut())
        };
        if !more_data(st) && !ok(st) {
            return None;
        }
        if size == 0 {
            return Some(0.0);
        }
        let mut buf = vec![0u8; size as usize];
        let st = unsafe {
            PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, buf.as_mut_ptr())
        };
        if !ok(st) {
            return None;
        }
        let needle = format!("pid_{pid}_");
        let mut sum = 0.0f64;
        let items = unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const PdhFmtItem, count as usize)
        };
        for item in items {
            if item.name.is_null() || item.value.c_status != 0 {
                continue;
            }
            let mut n = 0usize;
            while unsafe { *item.name.add(n) } != 0 {
                n += 1;
            }
            let name = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(item.name, n) })
                .to_ascii_lowercase();
            if name.contains(&needle) {
                sum += item.value.value.max(0.0);
            }
        }
        Some(sum)
    }

    /// Dedicated + shared GPU bytes for `pid` via WDDM VidMm (same source as Task Manager).
    pub fn process_gpu_bytes(pid: u32) -> Option<(f64, f64)> {
        let mut query: PdhHandle = 0;
        if !ok(unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut query) }) {
            return None;
        }
        let mut dedicated_h: PdhHandle = 0;
        let mut shared_h: PdhHandle = 0;
        let dedicated_path = wide("\\GPU Process Memory(*)\\Dedicated Usage");
        let shared_path = wide("\\GPU Process Memory(*)\\Shared Usage");
        let add_ok = unsafe {
            ok(PdhAddEnglishCounterW(query, dedicated_path.as_ptr(), 0, &mut dedicated_h))
                && ok(PdhAddEnglishCounterW(query, shared_path.as_ptr(), 0, &mut shared_h))
        };
        if !add_ok {
            unsafe { PdhCloseQuery(query) };
            return None;
        }
        if !ok(unsafe { PdhCollectQueryData(query) }) {
            unsafe { PdhCloseQuery(query) };
            return None;
        }
        std::thread::sleep(Duration::from_millis(120));
        if !ok(unsafe { PdhCollectQueryData(query) }) {
            unsafe { PdhCloseQuery(query) };
            return None;
        }
        let mut dedicated = sum_pid_bytes(dedicated_h, pid).unwrap_or(0.0);
        let mut shared = sum_pid_bytes(shared_h, pid).unwrap_or(0.0);
        if dedicated <= 0.0 && shared <= 0.0 {
            std::thread::sleep(Duration::from_millis(200));
            if ok(unsafe { PdhCollectQueryData(query) }) {
                dedicated = sum_pid_bytes(dedicated_h, pid).unwrap_or(0.0);
                shared = sum_pid_bytes(shared_h, pid).unwrap_or(shared);
            }
        }
        unsafe { PdhCloseQuery(query) };
        Some((dedicated, shared))
    }
}

/// README table: inference 150–300 MB, training ~2 GB.
///
/// Inference is dominated by the CUDA context on a WDDM GTX 1660 plus
/// `[x|y|z]` attention maps. Training adds Adam states and the unrolled
/// L×H autograd graph (attention scores retained per step).
pub fn build_vram_report(batch: usize, seq: usize, dim: usize, train_batch: usize) -> VramReport {
    let budget = MemoryBudget::default();
    let concat = seq.saturating_mul(3).max(1);
    let heads = 8usize;
    let n_l = 6.0f64;
    let n_sup = 16.0f64;
    let layers = 2.0f64;
    let cuda_context_mb = 200.0;
    let bytes = 4.0f64;

    let attn_mb = |b: usize| {
        (b * heads * concat * concat) as f64 * bytes / 1024.0 / 1024.0
    };
    let hidden_mb = |b: usize, bufs: usize| {
        budget.estimate_activation_mb(b, concat, dim, bufs)
    };
    let weight_fp32_mb = (1.58e6 * 4.0) / (1024.0 * 1024.0);
    let weight_ternary_mb = (1.58e6 * 2.0 / 8.0) / (1024.0 * 1024.0);

    let inference_est_mb = cuda_context_mb
        + weight_ternary_mb
        + hidden_mb(batch, 12)
        + attn_mb(batch)
        + 32.0;

    let training_est_mb = cuda_context_mb
        + weight_fp32_mb * 4.0 // param + grad + Adam m/v
        + n_l * n_sup * attn_mb(train_batch)
        + n_l * n_sup * layers * hidden_mb(train_batch, 2)
        + 64.0;

    VramReport {
        gpu: sample_nvidia_smi(),
        process: sample_this_process_vram(),
        budget: MemoryBudgetSnapshot::from(&budget),
        inference_est_mb,
        training_est_mb,
        readme_inference_mb: [150.0, 300.0],
        readme_training_mb: 2048.0,
    }
}

pub fn write_vram_report<P: AsRef<Path>>(path: P, report: &VramReport) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}
