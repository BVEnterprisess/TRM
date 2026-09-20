#[cfg(test)]
mod tests {
    use candle_core::Device;
    use trm_omega::network::{NetworkConfig, NetworkVariant};
    use trm_omega::recursion::{TrmConfig, TrmModel};
    use trm_omega::train::checkpoint::{save_checkpoint, load_checkpoint, TrainingMeta};
    use tempfile::tempdir;

    #[test]
    fn checkpoint_roundtrip() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let net_cfg = NetworkConfig {
            variant: NetworkVariant::Transformer,
            dim: 32,
            num_heads: 4,
            max_seq_len: 20,
            vocab_size: 11,
            num_layers: 2,
            ..Default::default()
        };
        let trm_cfg = TrmConfig::default();
        let mut model = TrmModel::new(device.clone(), net_cfg, trm_cfg)?;

        let dir = tempdir()?;

        let meta = TrainingMeta {
            epoch: 5,
            global_step: 100,
            best_val_acc: 0.95,
            best_val_loss: 0.1,
            rng_seed: 42,
            ema_decay: 0.999,
            ema_warmup: 1000,
            ema_step: 100,
            current_batch_size: 16,
        };

        use trm_omega::train::ema::EMA;
        let ema = EMA::new(0.999, 1000);

        save_checkpoint(dir.path(), "test", &model.varmap, &ema, &meta)?;

        let mut loaded_ema = EMA::new(0.999, 1000);
        let loaded_meta = load_checkpoint(dir.path(), "test", &mut model.varmap, &mut loaded_ema)?;

        assert_eq!(loaded_meta.epoch, meta.epoch);
        assert_eq!(loaded_meta.global_step, meta.global_step);
        assert_eq!(loaded_meta.best_val_acc, meta.best_val_acc);
        assert_eq!(loaded_meta.current_batch_size, meta.current_batch_size);

        Ok(())
    }
}
