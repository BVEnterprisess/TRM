#[cfg(test)]
mod tests {
    use candle_core::Device;
    use tempfile::tempdir;
    use trm_omega::data::load_arc_tasks;
    use trm_omega::network::{NetworkConfig, NetworkVariant};
    use trm_omega::recursion::{TrmConfig, TrmModel};
    use trm_omega::task_eval::evaluate_arc_tasks;

    #[test]
    fn eval_sample_tasks_returns_counts() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let tasks = load_arc_tasks(dir.path())?;
        assert!(!tasks.is_empty(), "sample ARC tasks should auto-seed");

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
            n_sup: 1,
            z_seq: 9,
            max_puzzles: 16,
            use_learned_halt_head: false,
            ..Default::default()
        };
        let model = TrmModel::new(device.clone(), net, trm)?;
        let result = evaluate_arc_tasks(&model, &device, &tasks, 1, false)?;
        assert!(result.total >= 4, "four seeded puzzles each have a test input");
        assert!(result.accuracy().is_finite());
        Ok(())
    }
}
