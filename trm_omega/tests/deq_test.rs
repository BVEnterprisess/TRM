#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor, Var};
    use trm_omega::deq::{solve_fixed_point, tensor_norm, DeqConfig, DeqWrapper};

    #[test]
    fn test_deq_converges() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8;

        // Construct a contractive linear/tanh operator: f(z) = tanh(0.5 * W * z + b)
        // ||0.5 * W|| < 1 ensures contraction
        let w_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i as f32 * 0.17).sin() * 0.4) / (dim as f32).sqrt())
            .collect();
        let w = Tensor::from_vec(w_data, (dim, dim), &dev)?;
        let b = Tensor::from_vec(vec![0.1f32; dim], dim, &dev)?;

        let f_fn = |z: &Tensor| -> Result<Tensor> {
            let lin = z.matmul(&w)?;
            let lin = lin.broadcast_add(&b)?;
            lin.tanh()
        };

        let z0 = Tensor::zeros((1, dim), candle_core::DType::F32, &dev)?;
        let mut cfg = DeqConfig::default();
        cfg.tolerance = 1e-6;
        cfg.max_iter = 40;

        let z_star = solve_fixed_point(&z0, f_fn, &cfg)?;

        // Verify ||f(z*) - z*|| < 1e-5
        let f_z_star = f_fn(&z_star)?;
        let residual = f_z_star.sub(&z_star)?;
        let res_norm = tensor_norm(&residual)?;

        println!("DEQ Convergence Test: residual norm = {:.4e}", res_norm);
        assert!(
            res_norm < 1e-5,
            "DEQ residual norm {:.4e} exceeds tolerance 1e-5",
            res_norm
        );

        Ok(())
    }

    #[test]
    fn test_neumann_backward_matches_unrolled_autograd() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let dim = 6;

        // Weight matrix with spectral radius < 1
        let w_raw: Vec<f32> = (0..dim * dim)
            .map(|i| ((i as f32 * 0.31).cos() * 0.35) / (dim as f32).sqrt())
            .collect();
        let b_raw = vec![0.05f32; dim];
        let target_raw: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let w_tensor = Tensor::from_vec(w_raw.clone(), (dim, dim), &dev)?;
        let b_tensor = Tensor::from_vec(b_raw.clone(), dim, &dev)?;
        let y_target = Tensor::from_vec(target_raw.clone(), (1, dim), &dev)?;

        // ─── 1. Unrolled Autograd Ground Truth (K = 60 steps) ──────────────
        let w_var_unroll = Var::from_tensor(&w_tensor)?;
        let b_var_unroll = Var::from_tensor(&b_tensor)?;

        let mut z_unroll = Tensor::zeros((1, dim), candle_core::DType::F32, &dev)?;
        let k_unroll = 60;
        for _ in 0..k_unroll {
            let lin = z_unroll.matmul(w_var_unroll.as_tensor())?;
            let lin = lin.broadcast_add(b_var_unroll.as_tensor())?;
            z_unroll = lin.tanh()?;
        }

        // Loss L = 0.5 * ||z_K - y_target||^2
        let diff_unroll = z_unroll.sub(&y_target)?;
        let loss_unroll = diff_unroll.sqr()?.sum_all()?.affine(0.5, 0.0)?;
        let grads_unroll = loss_unroll.backward()?;
        let grad_w_unroll = grads_unroll
            .get(w_var_unroll.as_tensor())
            .expect("grad_w_unroll must be computed");
        let grad_w_unroll_vec = grad_w_unroll.flatten_all()?.to_vec1::<f32>()?;

        // ─── 2. DEQ Fixed Point + Neumann Backward ────────────────────────
        let mut deq_cfg = DeqConfig::default();
        deq_cfg.tolerance = 1e-7;
        deq_cfg.max_iter = 50;
        deq_cfg.neumann_iters = 40;

        let deq = DeqWrapper::new(dev.clone(), deq_cfg);

        let f_fixed = |z: &Tensor| -> Result<Tensor> {
            let lin = z.matmul(&w_tensor)?;
            let lin = lin.broadcast_add(&b_tensor)?;
            lin.tanh()
        };

        let z0 = Tensor::zeros((1, dim), candle_core::DType::F32, &dev)?;
        let z_star = deq.forward(&z0, f_fixed)?;

        // Loss on z*: L = 0.5 * ||z* - y_target||^2
        // Gradient v = ∂L/∂z* = (z* - y_target)
        let v = z_star.sub(&y_target)?;

        // Neumann solve: u = (I - J^T)^(-1) * v
        let u = deq.backward(&z_star, &v, f_fixed)?;

        // Implicit parameter gradient:
        // ∂L/∂W = ∂(u . f(z*, W, b))/∂W
        let w_var_deq = Var::from_tensor(&w_tensor)?;
        let b_var_deq = Var::from_tensor(&b_tensor)?;

        let lin_deq = z_star.matmul(w_var_deq.as_tensor())?;
        let lin_deq = lin_deq.broadcast_add(b_var_deq.as_tensor())?;
        let f_at_zstar = lin_deq.tanh()?;

        let dot = (&u * &f_at_zstar)?.sum_all()?;
        let grads_deq = dot.backward()?;
        let grad_w_deq = grads_deq
            .get(w_var_deq.as_tensor())
            .expect("grad_w_deq must be computed");
        let grad_w_deq_vec = grad_w_deq.flatten_all()?.to_vec1::<f32>()?;

        // ─── 3. Verification: Compare Gradients to 1e-3 ─────────────────────
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;

        for (i, (&g_unroll, &g_deq)) in grad_w_unroll_vec.iter().zip(grad_w_deq_vec.iter()).enumerate() {
            let abs_diff = (g_unroll - g_deq).abs();
            let rel_diff = abs_diff / (g_unroll.abs().max(1e-6));
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
            println!(
                "  Param [{}]: unrolled = {:+.6}, deq_neumann = {:+.6}, abs_diff = {:.4e}",
                i, g_unroll, g_deq, abs_diff
            );
        }

        println!("============================================================");
        println!(" DEQ Neumann vs Unrolled Autograd Verification");
        println!(" Max Absolute Difference: {:.4e}", max_abs_diff);
        println!(" Max Relative Difference: {:.4e}", max_rel_diff);
        println!(" Target Tolerance       : 1.0000e-3");
        println!("============================================================");

        assert!(
            max_abs_diff < 1e-3,
            "Neumann backward gradient difference ({:.4e}) exceeds target tolerance 1e-3!",
            max_abs_diff
        );

        println!("✓ SUCCESS: Neumann backward matches unrolled autograd within 1e-3!");
        Ok(())
    }

    fn toy_contractive(dim: usize, dev: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let w_raw: Vec<f32> = (0..dim * dim)
            .map(|i| ((i as f32 * 0.31).cos() * 0.35) / (dim as f32).sqrt())
            .collect();
        let w = Tensor::from_vec(w_raw, (dim, dim), dev)?;
        let b = Tensor::from_vec(vec![0.05f32; dim], dim, dev)?;
        let y = Tensor::from_vec(
            (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect::<Vec<_>>(),
            (1, dim),
            dev,
        )?;
        Ok((w, b, y))
    }

    fn unroll_grad_w(w: &Tensor, b: &Tensor, y_target: &Tensor, dim: usize, k: usize) -> anyhow::Result<Vec<f32>> {
        let dev = w.device();
        let w_var = Var::from_tensor(w)?;
        let b_var = Var::from_tensor(b)?;
        let mut z = Tensor::zeros((1, dim), candle_core::DType::F32, &dev)?;
        for _ in 0..k {
            let lin = z.matmul(w_var.as_tensor())?;
            let lin = lin.broadcast_add(b_var.as_tensor())?;
            z = lin.tanh()?;
        }
        let loss = z.sub(y_target)?.sqr()?.sum_all()?.affine(0.5, 0.0)?;
        let grads = loss.backward()?;
        Ok(grads.get(w_var.as_tensor()).unwrap().flatten_all()?.to_vec1::<f32>()?)
    }

    fn neumann_grad_w(w: &Tensor, b: &Tensor, y_target: &Tensor, dim: usize, iters: usize) -> anyhow::Result<Vec<f32>> {
        let dev = w.device();
        let mut cfg = DeqConfig::default();
        cfg.tolerance = 1e-7;
        cfg.max_iter = 50;
        cfg.neumann_iters = iters;
        let deq = DeqWrapper::new(dev.clone(), cfg);
        let f_fixed = |z: &Tensor| -> Result<Tensor> {
            let lin = z.matmul(w)?;
            let lin = lin.broadcast_add(b)?;
            lin.tanh()
        };
        let z0 = Tensor::zeros((1, dim), candle_core::DType::F32, &dev)?;
        let z_star = deq.forward(&z0, f_fixed)?;
        let v = z_star.sub(y_target)?;
        let u = deq.backward(&z_star, &v, f_fixed)?;
        let w_var = Var::from_tensor(w)?;
        let b_var = Var::from_tensor(b)?;
        let lin = z_star.matmul(w_var.as_tensor())?;
        let f_at = lin.broadcast_add(b_var.as_tensor())?.tanh()?;
        let dot = (&u * &f_at)?.sum_all()?;
        let grads = dot.backward()?;
        Ok(grads.get(w_var.as_tensor()).unwrap().flatten_all()?.to_vec1::<f32>()?)
    }

    #[test]
    fn neumann_more_iters_reduces_error_vs_unroll() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let dim = 6;
        let (w, b, y) = toy_contractive(dim, &dev)?;
        let g_true = unroll_grad_w(&w, &b, &y, dim, 60)?;
        let err = |g: &[f32]| -> f32 {
            g.iter()
                .zip(g_true.iter())
                .map(|(a, t)| (a - t).abs())
                .fold(0.0f32, f32::max)
        };
        let e5 = err(&neumann_grad_w(&w, &b, &y, dim, 5)?);
        let e25 = err(&neumann_grad_w(&w, &b, &y, dim, 25)?);
        let e40 = err(&neumann_grad_w(&w, &b, &y, dim, 40)?);
        eprintln!("Neumann max-abs vs 60-unroll: 5={e5:.4e} 25={e25:.4e} 40={e40:.4e}");
        assert!(e40 < 1e-3, "40 iters should meet the 1e-3 verification bound, got {e40}");
        assert!(
            e5 + 1e-8 >= e40,
            "5 Neumann iters should not beat 40 (5={e5} 40={e40})"
        );
        let _ = e25;
        Ok(())
    }
}
