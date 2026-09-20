//! The 7M config is a named architecture, not folklore.
//! Instantiation must succeed on CPU and report a live VarMap count near 7M.

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use trm_omega::network::NetworkConfig;
    use trm_omega::preset::{self, Architecture};
    use trm_omega::recursion::{TrmConfig, TrmModel};

    fn net_from(arch: &Architecture) -> NetworkConfig {
        let mut cfg = NetworkConfig::default();
        arch.apply_network(&mut cfg);
        cfg.max_seq_len = 81 * 3;
        cfg
    }

    #[test]
    fn parse_7m_is_dim_448_heads_8_layers_2() {
        let a = preset::parse("7m").expect("7m preset");
        assert_eq!(a.name, "7m");
        assert_eq!(a.dim, 448);
        assert_eq!(a.heads, 8);
        assert_eq!(a.layers, 2);
        assert_eq!(a.vocab, 11);
        assert_eq!(a.dim % a.heads, 0);
    }

    #[test]
    fn parse_aliases() {
        assert_eq!(preset::parse("7M").unwrap().name, "7m");
        assert_eq!(preset::parse("paper").unwrap().dim, 256);
        assert_eq!(preset::parse("tiny").unwrap().dim, 32);
        assert!(preset::parse("not-a-preset").is_err());
    }

    #[test]
    fn seven_m_instantiates_and_counts_near_7m() -> anyhow::Result<()> {
        let arch = preset::seven_m();
        let net = net_from(&arch);
        let trm = TrmConfig {
            n_l_cycles: 6,
            n_sup: 16,
            z_seq: 81,
            max_puzzles: 4096,
            use_learned_halt_head: true,
            collect_intermediates: true,
            ..Default::default()
        };
        let model = TrmModel::new(Device::Cpu, net.clone(), trm)?;
        let n = model.param_count();
        let millions = n as f64 / 1e6;

        let artifacts = std::path::Path::new("artifacts");
        let _ = std::fs::create_dir_all(artifacts);
        let payload = serde_json::json!({
            "preset": arch.name,
            "dim": net.dim,
            "heads": net.num_heads,
            "layers": net.num_layers,
            "vocab": net.vocab_size,
            "ffn_hidden": net.ffn_hidden(),
            "max_puzzles": 4096,
            "live_params": n,
            "live_params_m": millions,
            "layer_est": net.param_count_estimate(),
        });
        std::fs::write(
            artifacts.join("preset_7m.json"),
            serde_json::to_vec_pretty(&payload)?,
        )?;

        assert_eq!(model.net_cfg.num_layers, 2);
        assert_eq!(
            n, 6_787_969,
            "7m live count changed: {n} ({millions:.3}M)"
        );
        eprintln!(
            "7m instantiated: dim={} heads={} layers={} live_params={} ({:.3}M)",
            net.dim, net.num_heads, net.num_layers, n, millions
        );
        Ok(())
    }

    #[test]
    fn paper_live_count_stays_2_67m() -> anyhow::Result<()> {
        let arch = preset::paper();
        let net = net_from(&arch);
        let trm = TrmConfig {
            max_puzzles: 4096,
            use_learned_halt_head: true,
            ..Default::default()
        };
        let n = TrmModel::new(Device::Cpu, net, trm)?.param_count();
        assert!(
            (2_600_000..=2_750_000).contains(&n),
            "paper live count drifted: {n}"
        );
        Ok(())
    }
}
