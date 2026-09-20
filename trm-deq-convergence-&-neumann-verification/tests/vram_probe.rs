use std::path::PathBuf;
use std::time::Instant;

use candle_core::{Device, Tensor};
use trm_omega::memory::{
    build_vram_report, write_vram_report, MemoryBudget, sample_nvidia_smi, sample_this_process_vram,
};
use trm_omega::network::{NetworkConfig, NetworkVariant};
use trm_omega::recursion::{TrmConfig, TrmModel};

#[test]
fn vram_and_throughput_harness() -> anyhow::Result<()> {
    let gpu = sample_nvidia_smi();
    if let Some(ref g) = gpu {
        eprintln!(
            "GPU {}  total={:.0} MiB used={:.0} MiB free={:.0} MiB",
            g.name, g.total_mb, g.used_mb, g.free_mb
        );
        assert!(
            g.total_mb >= 5000.0,
            "expected a 6GB-class card, got {:.0} MiB",
            g.total_mb
        );
    } else {
        eprintln!("nvidia-smi not available; writing estimate-only report");
    }

    // README numbers are for dim=256 / seq=81 on a GTX 1660. Estimates do not
    // require running that graph; a tiny packed forward still exercises the
    // kernel dispatch used by the server `--kernels` path.
    let mut report = build_vram_report(1, 81, 256, 8);
    let budget = MemoryBudget::default();
    assert!(
        report.inference_est_mb < 450.0,
        "inference estimate {:.1} MB is far above README 150–300 MB",
        report.inference_est_mb
    );
    assert!(
        report.training_est_mb < budget.usable_vram_mb as f64,
        "training estimate {:.1} MB exceeds usable VRAM {}",
        report.training_est_mb,
        budget.usable_vram_mb
    );

    let device = Device::Cpu;
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
        x_seq: 4,
        y_seq: 4,
        max_puzzles: 8,
        use_learned_halt_head: false,
        collect_intermediates: false,
        ..Default::default()
    };
    let mut model = TrmModel::new(device.clone(), net, trm)?;
    model.pack_ternary_inference()?;

    #[cfg(feature = "cuda")]
    {
        match trm_omega::custom_kernels::global_bank() {
            Some(bank) => {
                eprintln!(
                    "KernelBank live on GPU, kernels={:?}",
                    bank.loaded_kernels()
                );
                let _ = bank.dev.synchronize();
            }
            None => eprintln!("KernelBank failed to initialize"),
        }
    }

    let x = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device)?;
    let p = Tensor::from_vec(vec![0u32], 1, &device)?;
    let _ = model.forward_trace(&x, 4, Some(&p), None, None, None)?;
    let start = Instant::now();
    let iters = 8usize;
    for _ in 0..iters {
        let _ = model.forward_trace(&x, 4, Some(&p), None, None, None)?;
    }
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    let proc = sample_this_process_vram();
    if let Some(p) = &proc {
        eprintln!(
            "this process GPU memory {:.1} MiB via {}",
            p.used_mb, p.source
        );
        if let (Some(free), Some(total)) = (p.cuda_free_mb, p.cuda_total_mb) {
            eprintln!("cuda device {:.0} free / {:.0} total MiB", free, total);
        }
    }
    #[cfg(all(windows, feature = "cuda"))]
    {
        let p = proc.as_ref().expect("WDDM or CUDA process VRAM should be measurable after KernelBank init");
        assert!(
            p.used_mb > 1.0 || p.cuda_total_mb.unwrap_or(0.0) > 1000.0,
            "process VRAM still empty: {p:?}"
        );
        assert_ne!(p.source, "nvidia-smi", "WDDM nvidia-smi path should not be the source");
    }
    report.process = proc;

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("artifacts");
    path.push("vram_probe.json");
    write_vram_report(&path, &report)?;
    eprintln!(
        "tiny packed {:.2} iter/s  inf_est {:.1} MB  train_est {:.1} MB  packed {}  wrote {}",
        iters as f64 / secs,
        report.inference_est_mb,
        report.training_est_mb,
        model.packed_linear_count(),
        path.display()
    );
    Ok(())
}
