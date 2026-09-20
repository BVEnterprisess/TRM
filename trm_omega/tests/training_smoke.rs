#[cfg(test)]
mod tests {
    use candle_core::Device;
    use trm_omega::network::{NetworkConfig, NetworkVariant};
    use trm_omega::recursion::TrmConfig;
    use trm_omega::train::trainer::{TaskType, TrainConfig, Trainer};

    fn snapshot_params(trainer: &Trainer) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
        let data = trainer.model.varmap.data().lock().unwrap();
        let mut out = Vec::new();
        for (name, var) in data.iter() {
            out.push((name.clone(), var.as_tensor().flatten_all()?.to_vec1::<f32>()?));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn l2_param_delta(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> f32 {
        let mut acc = 0.0f32;
        for ((_, x), (_, y)) in a.iter().zip(b.iter()) {
            for (u, v) in x.iter().zip(y.iter()) {
                let d = u - v;
                acc += d * d;
            }
        }
        acc.sqrt()
    }

    #[test]
    fn training_smoke_test_updates_weights() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let net_cfg = NetworkConfig {
            variant: NetworkVariant::Transformer,
            dim: 32,
            num_heads: 4,
            max_seq_len: 48,
            vocab_size: 11,
            num_layers: 2,
            ..Default::default()
        };
        let trm_cfg = TrmConfig {
            n_l_cycles: 2,
            n_sup: 2,
            collect_intermediates: true,
            z_seq: 8,
            max_puzzles: 32,
            use_learned_halt_head: false,
            ..Default::default()
        };
        let train_cfg = TrainConfig {
            data_dir: ".".into(),
            task_type: TaskType::Arc,
            batch_size: 2,
            micro_batch_size: 2,
            max_epochs: 1,
            eval_every_epochs: 100,
            save_every_epochs: 100,
            log_every_steps: 1,
            seed: 42,
            no_augment: true,
            use_fp16: false,
            warmup_steps: 0,
            weight_decay: 0.0,
            puzzle_wd: 0.0,
            halt_lambda: 0.0,
            ..Default::default()
        };

        let mut fake_data = Vec::new();
        for i in 0..8 {
            let x: Vec<u32> = (0..8).map(|_| rand::random::<u32>() % 10).collect();
            let y: Vec<u32> = (0..8).map(|_| rand::random::<u32>() % 10).collect();
            fake_data.push(trm_omega::data::TokenizedExample {
                puzzle_id: i,
                x_tokens: x.clone(),
                y_tokens: y.clone(),
                rows: 1,
                cols: 8,
                x_seq_len: x.len(),
                y_seq_len: y.len(),
            });
        }

        let mut trainer = Trainer::new(device, net_cfg, trm_cfg, train_cfg)?;
        let before = snapshot_params(&trainer)?;
        let loss = trainer.train_epoch(&fake_data)?;
        let after = snapshot_params(&trainer)?;

        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let delta = l2_param_delta(&before, &after);
        assert!(
            delta > 1e-6,
            "optimizer must move weights; L2 delta was {delta}"
        );
        Ok(())
    }
}
