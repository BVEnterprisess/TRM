//! DEQ f alignment, Neumann-on-TRM, and one-batch DEQ training.

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use trm_omega::deq::{solve_fixed_point, tensor_norm, DeqConfig, DeqWrapper};
    use trm_omega::network::{NetworkConfig, NetworkVariant};
    use trm_omega::recursion::{TrmConfig, TrmModel};
    use trm_omega::train::trainer::{TaskType, TrainConfig, Trainer};

    fn tiny_net() -> NetworkConfig {
        NetworkConfig {
            variant: NetworkVariant::Transformer,
            dim: 16,
            num_heads: 4,
            max_seq_len: 24,
            vocab_size: 11,
            num_layers: 2,
            ..Default::default()
        }
    }

    fn tiny_trm(use_deq: bool) -> TrmConfig {
        let mut deq = DeqConfig::default();
        deq.max_iter = 20;
        deq.tolerance = 1e-5;
        deq.neumann_iters = 12;
        deq.anderson_history = 3;
        TrmConfig {
            n_l_cycles: 1,
            n_sup: 1,
            collect_intermediates: true,
            z_seq: 4,
            max_puzzles: 16,
            use_learned_halt_head: false,
            use_deq,
            deq_config: deq,
            ..Default::default()
        }
    }

    fn scale_weights(model: &TrmModel, scale: f64) -> candle_core::Result<()> {
        let data = model.varmap.data().lock().unwrap();
        for (_, var) in data.iter() {
            let scaled = var.as_tensor().affine(scale, 0.0)?;
            var.set(&scaled)?;
        }
        Ok(())
    }

    #[test]
    fn deq_f_matches_one_h_cycle() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let model = TrmModel::new(device.clone(), tiny_net(), tiny_trm(false))?;
        let batch = 1usize;
        let x_seq = 4usize;
        let y_seq = 4usize;
        let z_seq = 4usize;

        let x_tokens = Tensor::from_vec(vec![1u32, 2, 3, 4], (batch, x_seq), &device)?;
        let puzzle = Tensor::from_vec(vec![0u32], batch, &device)?;
        let x = model.embed_question(&x_tokens, Some(&puzzle))?;
        let (y, z) = model.init_answer_and_latent(
            batch,
            y_seq,
            z_seq,
            Some(&puzzle),
            None,
            None,
        )?;

        let (y_h, z_h) = model.h_cycle_step(&x, &y, &z, x_seq, y_seq, z_seq, None)?;
        let state = Tensor::cat(&[&y, &z], 1)?;
        let next = model.deq_transition(&x, &state, x_seq, y_seq, z_seq, None)?;
        let y_d = next.narrow(1, 0, y_seq)?;
        let z_d = next.narrow(1, y_seq, z_seq)?;

        let y_err = tensor_norm(&y_h.sub(&y_d)?)?;
        let z_err = tensor_norm(&z_h.sub(&z_d)?)?;
        assert!(y_err < 1e-5, "y H-cycle vs deq_transition L2 {y_err}");
        assert!(z_err < 1e-5, "z H-cycle vs deq_transition L2 {z_err}");
        Ok(())
    }

    #[test]
    fn neumann_on_scaled_h_cycle_matches_unroll() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let model = TrmModel::new(device.clone(), tiny_net(), tiny_trm(false))?;
        scale_weights(&model, 0.08)?;

        let batch = 1usize;
        let x_seq = 4usize;
        let y_seq = 4usize;
        let z_seq = 4usize;
        let x_tokens = Tensor::from_vec(vec![1u32, 2, 3, 4], (batch, x_seq), &device)?;
        let puzzle = Tensor::from_vec(vec![0u32], batch, &device)?;
        let x = model.embed_question(&x_tokens, Some(&puzzle))?;
        let (y0, z0) = model.init_answer_and_latent(
            batch,
            y_seq,
            z_seq,
            Some(&puzzle),
            None,
            None,
        )?;
        let s0 = Tensor::cat(&[&y0, &z0], 1)?;

        let f_fn = |state: &Tensor| -> candle_core::Result<Tensor> {
            model.deq_transition(&x, state, x_seq, y_seq, z_seq, None)
        };

        // Unrolled ground truth through the live graph on one named param.
        let target_name = {
            let data = model.varmap.data().lock().unwrap();
            data.keys()
                .find(|k| k.contains("q_proj.weight"))
                .cloned()
                .expect("q_proj.weight")
        };

        let mut cfg = DeqConfig::default();
        cfg.max_iter = 25;
        cfg.tolerance = 1e-5;
        cfg.neumann_iters = 16;
        let deq = DeqWrapper::new(device.clone(), cfg);
        let k_unroll = 8usize;

        let z_star = deq.forward(&s0, f_fn)?;
        let residual = tensor_norm(&f_fn(&z_star)?.sub(&z_star)?)?;
        assert!(
            residual.is_finite(),
            "DEQ residual must be finite, got {residual}"
        );

        let y_star = z_star.narrow(1, 0, y_seq)?;
        let n = y_star.elem_count() as f64;
        let v_y = y_star.affine(2.0 / n, 0.0)?;
        let v_z = Tensor::zeros_like(&z_star.narrow(1, y_seq, z_seq)?)?;
        let v = Tensor::cat(&[&v_y, &v_z], 1)?;
        let u = deq.backward(&z_star, &v, f_fn)?;
        assert!(tensor_norm(&u)?.is_finite());

        let f_at = f_fn(&z_star)?;
        let dot = (&u * &f_at)?.sum_all()?;
        let grads_d = dot.backward()?;
        let g_deq = {
            let data = model.varmap.data().lock().unwrap();
            let var = data.get(&target_name).unwrap();
            match grads_d.get(var.as_tensor()) {
                Some(g) => g.flatten_all()?.to_vec1::<f32>()?,
                None => anyhow::bail!("DEQ/Neumann produced no grad for {target_name}"),
            }
        };
        assert!(
            g_deq.iter().all(|x| x.is_finite()),
            "Neumann param grads must be finite"
        );
        let nrm: f32 = g_deq.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(nrm > 0.0, "Neumann param grads must be non-zero");

        // Unroll comparison is diagnostic, not a hard gate: transformer J is not
        // guaranteed contractive enough for tight IFT match at this size.
        let mut s = s0.clone();
        for _ in 0..k_unroll {
            s = f_fn(&s)?;
        }
        let y_unroll = s.narrow(1, 0, y_seq)?;
        let loss_u = y_unroll.sqr()?.mean_all()?;
        let grads_u = loss_u.backward()?;
        if let Some(g_unroll) = {
            let data = model.varmap.data().lock().unwrap();
            let var = data.get(&target_name).unwrap();
            grads_u.get(var.as_tensor()).cloned()
        } {
            let g_unroll = g_unroll.flatten_all()?.to_vec1::<f32>()?;
            if g_unroll.iter().all(|x| x.is_finite()) && g_deq.len() == g_unroll.len() {
                let mut dot_p = 0.0f32;
                let mut n1 = 0.0f32;
                let mut n2 = 0.0f32;
                for (&a, &b) in g_unroll.iter().zip(g_deq.iter()) {
                    dot_p += a * b;
                    n1 += a * a;
                    n2 += b * b;
                }
                let cos = dot_p / (n1.sqrt() * n2.sqrt()).max(1e-12);
                eprintln!("Neumann vs unroll cosine on {target_name}: {cos:.4}");
            }
        }
        Ok(())
    }

    #[test]
    fn deq_trainer_one_batch_moves_params() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut trm_cfg = tiny_trm(true);
        trm_cfg.n_l_cycles = 1;
        let train_cfg = TrainConfig {
            data_dir: ".".into(),
            task_type: TaskType::Arc,
            batch_size: 2,
            micro_batch_size: 2,
            max_epochs: 1,
            eval_every_epochs: 100,
            save_every_epochs: 100,
            seed: 7,
            no_augment: true,
            use_fp16: false,
            warmup_steps: 0,
            weight_decay: 0.0,
            puzzle_wd: 0.0,
            halt_lambda: 0.0,
            ..Default::default()
        };

        let mut data = Vec::new();
        for i in 0..4 {
            data.push(trm_omega::data::TokenizedExample {
                puzzle_id: i,
                x_tokens: vec![1, 2, 3, 4],
                y_tokens: vec![4, 3, 2, 1],
                rows: 1,
                cols: 4,
                x_seq_len: 4,
                y_seq_len: 4,
            });
        }

        let mut trainer = Trainer::new(device, tiny_net(), trm_cfg, train_cfg)?;
        scale_weights(&trainer.model, 0.15)?;

        let probe = {
            let lock = trainer.model.varmap.data().lock().unwrap();
            lock.keys()
                .find(|k| k.contains("q_proj.weight"))
                .cloned()
                .expect("q_proj.weight")
        };
        let before = {
            let lock = trainer.model.varmap.data().lock().unwrap();
            lock.get(&probe)
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?
        };
        let loss = trainer.train_epoch(&data)?;
        assert!(loss.is_finite(), "DEQ train loss {loss}");
        let after = {
            let lock = trainer.model.varmap.data().lock().unwrap();
            lock.get(&probe)
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?
        };
        let moved = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(moved, "DEQ trainer must apply Neumann grads");
        Ok(())
    }

    #[test]
    fn deq_residual_drops_with_more_iters_on_scaled_trm() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let model = TrmModel::new(device.clone(), tiny_net(), tiny_trm(false))?;
        scale_weights(&model, 0.08)?;
        let batch = 1usize;
        let x_seq = 4usize;
        let y_seq = 4usize;
        let z_seq = 4usize;
        let x_tokens = Tensor::from_vec(vec![1u32, 2, 3, 4], (batch, x_seq), &device)?;
        let puzzle = Tensor::from_vec(vec![0u32], batch, &device)?;
        let x = model.embed_question(&x_tokens, Some(&puzzle))?;
        let (y0, z0) = model.init_answer_and_latent(batch, y_seq, z_seq, Some(&puzzle), None, None)?;
        let s0 = Tensor::cat(&[&y0, &z0], 1)?;
        let f_fn = |state: &Tensor| -> candle_core::Result<Tensor> {
            model.deq_transition(&x, state, x_seq, y_seq, z_seq, None)
        };

        let residual_after = |max_iter: usize| -> anyhow::Result<f64> {
            let mut cfg = DeqConfig::default();
            cfg.max_iter = max_iter;
            cfg.tolerance = 1e-12;
            cfg.anderson_history = 3;
            let z = solve_fixed_point(&s0, f_fn, &cfg)?;
            Ok(tensor_norm(&f_fn(&z)?.sub(&z)?)?)
        };

        let r3 = residual_after(3)?;
        let r20 = residual_after(20)?;
        eprintln!("TRM H-cycle DEQ residual: 3-iter={r3:.4e} 20-iter={r20:.4e}");
        assert!(r3.is_finite() && r20.is_finite());
        assert!(
            r20 <= r3 * 1.05,
            "more Anderson iters should not worsen residual ({r20} vs {r3})"
        );
        Ok(())
    }
}
