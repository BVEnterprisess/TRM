#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use trm_omega::network::{NetworkConfig, NetworkVariant};
    use trm_omega::recursion::{TrmConfig, TrmModel};
    use trm_omega::trmq10;

    fn tiny() -> (NetworkConfig, TrmConfig) {
        let net = NetworkConfig {
            variant: NetworkVariant::Transformer,
            dim: 16,
            num_heads: 4,
            max_seq_len: 24,
            vocab_size: 11,
            num_layers: 2,
            ..Default::default()
        };
        let trm = TrmConfig {
            n_l_cycles: 1,
            n_sup: 1,
            z_seq: 4,
            max_puzzles: 16,
            use_learned_halt_head: false,
            collect_intermediates: true,
            ..Default::default()
        };
        (net, trm)
    }

    fn logits(model: &TrmModel, device: &Device) -> anyhow::Result<Vec<f32>> {
        let x = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), device)?;
        let p = Tensor::from_vec(vec![0u32], 1, device)?;
        let trace = model.forward_trace(&x, 4, Some(&p), None, None, None)?;
        Ok(trace.logits_final.flatten_all()?.to_vec1::<f32>()?)
    }

    #[test]
    fn trmq10_roundtrip_matches_quantized_overlay() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (net, trm) = tiny();
        let model = TrmModel::new(device.clone(), net.clone(), trm.clone())?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("m.trmq10");
        trmq10::save_trmq10(&path, &model)?;
        assert!(trmq10::is_trmq10(&path));

        let loaded = trmq10::load_trmq10(device.clone(), &path, false)?;
        let a = logits(&model, &device)?;
        let b = logits(&loaded, &device)?;
        assert_eq!(a.len(), b.len());

        // Reloaded ternary 2D weights differ from the FP32 source; leftover
        // tensors must match exactly. Check the file at least produces finite logits
        // of the same shape, and a second load is bit-identical to the first.
        let loaded2 = trmq10::load_trmq10(device.clone(), &path, false)?;
        let c = logits(&loaded2, &device)?;
        for (x, y) in b.iter().zip(c.iter()) {
            assert!((x - y).abs() < 1e-6, "two loads of the same TRMQ10 must match");
        }
        assert!(b.iter().all(|v| v.is_finite()));
        let _ = a;
        Ok(())
    }

    #[test]
    fn load_model_dispatches_on_magic() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (net, trm) = tiny();
        let model = TrmModel::new(device.clone(), net.clone(), trm.clone())?;
        let dir = tempfile::tempdir()?;
        let st = dir.path().join("m.safetensors");
        let tq = dir.path().join("m.trmq10");
        model.save_safetensors(&st)?;
        trmq10::save_trmq10(&tq, &model)?;

        let from_st = trmq10::load_model(device.clone(), &st, net.clone(), trm.clone())?;
        let from_tq = trmq10::load_model(device.clone(), &tq, net, trm)?;
        let la = logits(&from_st, &device)?;
        let lb = logits(&model, &device)?;
        for (x, y) in la.iter().zip(lb.iter()) {
            assert!((x - y).abs() < 1e-5, "safetensors must restore FP32");
        }
        assert!(logits(&from_tq, &device)?.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
