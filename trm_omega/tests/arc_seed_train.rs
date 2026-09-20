#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use candle_core::Device;
    use tempfile::tempdir;
    use trm_omega::data::{arc_training_examples, load_arc_tasks};
    use trm_omega::network::{NetworkConfig, NetworkVariant};
    use trm_omega::recursion::TrmConfig;
    use trm_omega::train::trainer::{TaskType, TrainConfig, Trainer};

    #[test]
    fn seeded_arc_train_writes_curve_and_loss_is_finite() -> anyhow::Result<()> {
        let data_dir = tempdir()?;
        let tasks = load_arc_tasks(data_dir.path())?;
        let train = arc_training_examples(&tasks);
        assert!(!train.is_empty());

        let device = Device::Cpu;
        let net = NetworkConfig {
            variant: NetworkVariant::Transformer,
            dim: 16,
            num_heads: 4,
            max_seq_len: 48,
            vocab_size: 11,
            num_layers: 2,
            ..Default::default()
        };
        let trm = TrmConfig {
            n_l_cycles: 1,
            n_sup: 2,
            z_seq: 9,
            max_puzzles: 16,
            use_learned_halt_head: false,
            collect_intermediates: true,
            ..Default::default()
        };
        let cfg = TrainConfig {
            data_dir: data_dir.path().to_path_buf(),
            task_type: TaskType::Arc,
            batch_size: 2,
            micro_batch_size: 2,
            max_epochs: 3,
            eval_every_epochs: 100,
            save_every_epochs: 100,
            seed: 42,
            no_augment: true,
            use_fp16: false,
            warmup_steps: 0,
            weight_decay: 0.0,
            puzzle_wd: 0.0,
            halt_lambda: 0.0,
            ..Default::default()
        };

        let mut trainer = Trainer::new(device, net, trm, cfg)?;
        let mut losses = Vec::new();
        for epoch in 0..3 {
            let loss = trainer.train_epoch(&train)?;
            assert!(loss.is_finite(), "epoch {epoch} loss {loss}");
            losses.push(loss);
        }

        let artifacts = std::path::Path::new("artifacts");
        let _ = fs::create_dir_all(artifacts);
        let csv_path = artifacts.join("arc_smoke.csv");
        let mut f = fs::File::create(&csv_path)?;
        writeln!(f, "epoch,loss")?;
        for (i, loss) in losses.iter().enumerate() {
            writeln!(f, "{},{}", i, loss)?;
        }

        eprintln!("seeded ARC losses: {:?}", losses);
        assert!(
            losses.iter().all(|l| l.is_finite() && *l >= 0.0),
            "training losses must be finite and non-negative"
        );
        Ok(())
    }
}
