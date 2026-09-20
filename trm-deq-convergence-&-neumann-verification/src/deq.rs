//! # Deep Equilibrium Module – 7M‑Agent Ready
//!
//! - Forward: Anderson acceleration using **mean residual** (O(1) w.r.t batch size).
//! - Backward: Implicit Function Theorem via Neumann series with dynamic residual check.
//! - Damping prevents divergence on 2‑bit quantised weights.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, Var};
use std::collections::VecDeque;

// ─── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeqConfig {
    pub max_iter: usize,
    pub tolerance: f64,
    pub anderson_history: usize,
    pub use_ift: bool,
    pub neumann_iters: usize,   // maximum Neumann iterations
    pub damping_base: f64,      // 0.7 typical, adjusted by residual
}

impl Default for DeqConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tolerance: 1e-6,
            anderson_history: 5,
            use_ift: true,
            neumann_iters: 25,
            damping_base: 0.7,
        }
    }
}

pub fn tensor_norm(t: &Tensor) -> candle_core::Result<f64> {
    let sq = t.to_dtype(candle_core::DType::F32)?.sqr()?.sum_all()?;
    let norm = sq.sqrt()?.to_scalar::<f32>()? as f64;
    Ok(norm)
}

// ─── Anderson State ──────────────────────────────────────────────────────

pub struct AndersonState {
    f_hist: VecDeque<Tensor>,
    z_hist: VecDeque<Tensor>,
    residuals: VecDeque<Tensor>,
    max_hist: usize,
}

impl AndersonState {
    pub fn new(max_hist: usize) -> Self {
        Self {
            f_hist: VecDeque::with_capacity(max_hist),
            z_hist: VecDeque::with_capacity(max_hist),
            residuals: VecDeque::with_capacity(max_hist),
            max_hist,
        }
    }

    pub fn push(&mut self, z: &Tensor, fz: &Tensor) -> candle_core::Result<()> {
        let residual = fz.sub(z)?;
        if self.z_hist.len() >= self.max_hist {
            self.z_hist.pop_front();
            self.f_hist.pop_front();
            self.residuals.pop_front();
        }
        self.z_hist.push_back(z.clone());
        self.f_hist.push_back(fz.clone());
        self.residuals.push_back(residual);
        Ok(())
    }

    /// Accelerate using the **mean residual** across the batch dimension.
    /// This keeps memory O(1) w.r.t batch size (critical for 7M agents).
    pub fn accelerate(&self, z: &Tensor, fz: &Tensor, damping: f64) -> candle_core::Result<Tensor> {
        if self.z_hist.is_empty() {
            return Ok(fz.clone());
        }

        let m = self.residuals.len();
        if m < 2 {
            return Ok(fz.clone());
        }

        // 1. Compute mean residual per feature dimension (over all batch & seq)
        let mut mean_residuals = Vec::with_capacity(m);
        for r in self.residuals.iter() {
            let dims = r.dims();
            let num_dims = dims.len();
            let r_flat = if num_dims > 1 {
                r.flatten(0, num_dims - 2)?
            } else {
                r.clone()
            };
            let r_mean = if r_flat.dims().len() > 1 {
                r_flat.mean(0)?
            } else {
                r_flat
            };
            mean_residuals.push(r_mean);
        }

        // 2. Build Gram matrix: [m, D] @ [D, m] -> [m, m]
        let res_stack = Tensor::stack(&mean_residuals, 0)?; // [m, D]
        let gram = res_stack.matmul(&res_stack.t()?)?;      // [m, m]

        // 3. Move only the tiny [m,m] to CPU for linear solve
        let gram_cpu = gram.to_device(&Device::Cpu)?;
        let a = gram_cpu.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        // 4. Solve K * alpha = [0,...,0,1]^T (size m+1) with Gaussian elimination
        let n = m + 1;
        let mut k = vec![0.0f64; n * n];
        for i in 0..m {
            for j in 0..m {
                k[i * n + j] = (a[i * m + j] as f64) + if i == j { 1e-6 } else { 0.0 };
            }
            k[i * n + m] = 1.0;
            k[m * n + i] = 1.0;
        }
        k[m * n + m] = 0.0;

        let mut mat = k.clone();
        let mut rhs = vec![0.0f64; n];
        rhs[m] = 1.0;

        for col in 0..n {
            let mut max_row = col;
            for row in col + 1..n {
                if mat[row * n + col].abs() > mat[max_row * n + col].abs() {
                    max_row = row;
                }
            }
            if max_row != col {
                for c in 0..n {
                    mat.swap(col * n + c, max_row * n + c);
                }
                rhs.swap(col, max_row);
            }
            let pivot = mat[col * n + col];
            if pivot.abs() < 1e-14 {
                continue;
            }
            for row in col + 1..n {
                let factor = mat[row * n + col] / pivot;
                for c in col..n {
                    mat[row * n + c] -= factor * mat[col * n + c];
                }
                rhs[row] -= factor * rhs[col];
            }
        }

        let mut alpha = vec![0.0f64; m];
        for i in (0..m).rev() {
            let mut sum = 0.0f64;
            for j in i + 1..m {
                sum += mat[i * n + j] * alpha[j];
            }
            let p = mat[i * n + i];
            if p.abs() > 1e-14 {
                alpha[i] = (rhs[i] - sum) / p;
            } else {
                alpha[i] = 1.0 / (m as f64);
            }
        }

        // Sanity check alpha
        let sum_alpha: f64 = alpha.iter().sum();
        if !sum_alpha.is_finite() || sum_alpha.abs() < 1e-4 {
            return Ok(fz.clone());
        }

        // 5. Anderson extrapolated state = sum alpha_i * F(z_i)
        let mut z_anderson = fz.clone();
        for i in 0..m {
            let scaled = self.f_hist[i].affine(alpha[i], 0.0)?;
            if i == 0 {
                z_anderson = scaled;
            } else {
                z_anderson = z_anderson.add(&scaled)?;
            }
        }

        // 6. Damped update: z_new = (1-d)*z + d*z_anderson
        let z_new = (z.affine(1.0 - damping, 0.0)? + z_anderson.affine(damping, 0.0)?)?;
        Ok(z_new)
    }
}

// ─── Forward Solver ──────────────────────────────────────────────────────

pub fn solve_fixed_point<F>(
    initial_z: &Tensor,
    f_fn: F,
    config: &DeqConfig,
) -> candle_core::Result<Tensor>
where
    F: Fn(&Tensor) -> candle_core::Result<Tensor>,
{
    let mut z = initial_z.clone();
    let mut state = AndersonState::new(config.anderson_history);

    for iter in 0..config.max_iter {
        let fz = f_fn(&z)?;
        let residual = fz.sub(&z)?;
        let norm_val = tensor_norm(&residual)?;

        if norm_val < config.tolerance {
            log::debug!("DEQ converged in {} iterations (norm = {:.2e})", iter + 1, norm_val);
            return Ok(fz);
        }

        state.push(&z, &fz)?;

        // Adaptive damping
        let damping = config.damping_base
            * (1.0 / (1.0 + norm_val * 5.0)).clamp(0.4, 0.95);
        let z_new = state.accelerate(&z, &fz, damping)?;
        z = z_new;
    }

    log::warn!("DEQ reached max_iter ({})", config.max_iter);
    f_fn(&z)
}

// ─── Neumann Solver with dynamic residual check ────────────────────────

pub fn neumann_solve<F>(
    v: &Tensor,
    z_star: &Tensor,
    f_fn: F,
    max_iters: usize,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> candle_core::Result<Tensor>,
{
    let mut u = v.clone();

    for k in 0..max_iters {
        let z_var = Var::from_tensor(z_star)
            .context("failed to create Var for z_star")?;
        let fz = f_fn(z_var.as_tensor())
            .context("failed evaluating f_fn in neumann step")?;
        
        let dot = (&u * &fz)?.sum_all()
            .context("failed computing dot product in neumann step")?;

        let grads = dot.backward()
            .context("backward failed in neumann step")?;
        let grad_z = grads.get(z_var.as_tensor())
            .context("grad_z not found in grad store")?;

        let u_next = (v + grad_z)?;

        // Check residual of the linear system: (I - J^T) u - v
        let diff = u_next.sub(&u)?;
        let nd = tensor_norm(&diff)?;

        u = u_next;

        if nd < 1e-4 {
            log::debug!("Neumann converged at iteration {} (res = {:.2e})", k + 1, nd);
            break;
        }
    }
    Ok(u)
}

// ─── Wrapper ─────────────────────────────────────────────────────────────

pub struct DeqWrapper {
    pub config: DeqConfig,
    pub device: Device,
}

impl DeqWrapper {
    pub fn new(device: Device, config: DeqConfig) -> Self {
        Self { config, device }
    }

    pub fn forward<F>(&self, initial_z: &Tensor, f_fn: F) -> candle_core::Result<Tensor>
    where
        F: Fn(&Tensor) -> candle_core::Result<Tensor>,
    {
        solve_fixed_point(initial_z, f_fn, &self.config)
    }

    pub fn backward<F>(
        &self,
        z_star: &Tensor,
        grad_output: &Tensor,
        f_fn: F,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor) -> candle_core::Result<Tensor>,
    {
        if !self.config.use_ift {
            return Ok(grad_output.clone());
        }
        neumann_solve(grad_output, z_star, f_fn, self.config.neumann_iters)
    }
}
