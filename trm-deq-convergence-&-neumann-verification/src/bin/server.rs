use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use candle_core::Tensor;
use trm_omega::default_device;
use trm_omega::network::{NetworkConfig, NetworkVariant};
use trm_omega::recursion::{TrmConfig, TrmModel};

#[derive(Parser, Debug)]
#[command(about = "TRM-Omega v10 inference server")]
struct Args {
    #[arg(short, long)]
    model: Option<String>,

    #[arg(long, default_value_t = 32)]
    batch: usize,

    #[arg(long, default_value_t = 81)]
    seq_len: usize,

    #[arg(long, default_value_t = 100)]
    iters: usize,

    #[arg(long, default_value_t = 256)]
    dim: usize,

    #[arg(long, default_value_t = 8)]
    heads: usize,

    #[arg(long, default_value = "transformer")]
    variant: String,

    #[arg(long, default_value_t = 11)]
    vocab: usize,

    #[arg(long, default_value_t = 2)]
    layers: usize,

    #[arg(long, default_value_t = false)]
    adaptive_halt: bool,

    #[arg(long, default_value_t = false)]
    deq: bool,

    #[arg(long, default_value_t = false)]
    auto_install: bool,

    /// Quantize live 2D linears and run them through packed ternary kernels.
    #[arg(long, default_value_t = false)]
    kernels: bool,

    #[arg(long, default_value_t = 6)]
    n_l: usize,

    #[arg(long, default_value_t = 16)]
    n_sup: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let _ = trm_omega::setup::ensure_dependencies(args.auto_install);

    let device = default_device()?;
    let variant = match args.variant.as_str() {
        "mixer" => NetworkVariant::MlpMixer,
        _ => NetworkVariant::Transformer,
    };

    let net_cfg = NetworkConfig {
        variant,
        dim: args.dim,
        num_heads: args.heads,
        max_seq_len: args.seq_len * 3,
        vocab_size: args.vocab,
        num_layers: args.layers,
        ..Default::default()
    };
    let trm_cfg = TrmConfig {
        use_adaptive_halt: args.adaptive_halt,
        use_deq: args.deq,
        n_l_cycles: args.n_l,
        n_sup: args.n_sup,
        ..Default::default()
    };

    let mut model = if let Some(path) = &args.model {
        trm_omega::trmq10::load_model(device.clone(), path, net_cfg.clone(), trm_cfg.clone())?
    } else {
        TrmModel::new(device.clone(), net_cfg.clone(), trm_cfg.clone())?
    };
    if args.kernels && model.packed_linear_count() == 0 {
        model.pack_ternary_inference()?;
    }

    let x_tokens = Tensor::zeros((args.batch, args.seq_len), candle_core::DType::U32, &device)?;
    let puzzle_ids = Tensor::zeros(args.batch, candle_core::DType::U32, &device)?;

    // warmup (also uploads packed weights into the device cache)
    let _ = model.forward_trace(&x_tokens, args.seq_len, Some(&puzzle_ids), None, None, None)?;
    let vram_after_warmup = trm_omega::memory::sample_nvidia_smi();

    let start = Instant::now();
    let mut steps_sum = 0usize;
    for _ in 0..args.iters {
        let trace = model.forward_trace(&x_tokens, args.seq_len, Some(&puzzle_ids), None, None, None)?;
        steps_sum += trace.steps_used;
    }
    let secs = start.elapsed().as_secs_f64();
    let vram_after = trm_omega::memory::sample_nvidia_smi();

    let trace = model.forward_trace(&x_tokens, args.seq_len, Some(&puzzle_ids), None, None, None)?;
    let preds = trace.preds_final.flatten_all()?.to_vec1::<u32>()?;
    let checksum: u64 = preds.iter().take(128).map(|&v| v as u64).sum();

    eprintln!("============================================================");
    eprintln!(" TRM-Omega v10.2 Inference Benchmark");
    eprintln!(" variant        : {:?}", net_cfg.variant);
    eprintln!(" dim / heads    : {} / {}", net_cfg.dim, net_cfg.num_heads);
    eprintln!(" batch / seq    : {} / {}", args.batch, args.seq_len);
    eprintln!(" n_l / n_sup    : {} / {}", args.n_l, args.n_sup);
    eprintln!(" iters          : {}", args.iters);
    eprintln!(" total secs     : {:.4}", secs);
    eprintln!(" iter/s         : {:.2}", args.iters as f64 / secs);
    eprintln!(" avg steps used : {:.2}", steps_sum as f64 / args.iters as f64);
    eprintln!(" checksum       : {}", checksum);
    eprintln!(" param est.     : {:.2}M (layers only)", net_cfg.param_count_estimate() as f64 / 1e6);
    eprintln!(" params live    : {:.2}M", model.param_count() as f64 / 1e6);
    eprintln!(" packed linears : {}", model.packed_linear_count());
    eprintln!(" kernel backend : {:?}", trm_omega::kernel_dispatch::active_backend());
    let report = trm_omega::memory::build_vram_report(args.batch, args.seq_len, args.dim, 8);
    eprintln!(
        " VRAM est inf   : {:.1} MB (README {:.0}–{:.0})",
        report.inference_est_mb, report.readme_inference_mb[0], report.readme_inference_mb[1]
    );
    eprintln!(
        " VRAM est train : {:.1} MB (README ~{:.0})",
        report.training_est_mb, report.readme_training_mb
    );
    let gpu = vram_after.as_ref().or(vram_after_warmup.as_ref()).or(report.gpu.as_ref());
    if let Some(g) = gpu {
        eprintln!(
            " nvidia-smi     : {} used {:.0}/{:.0} MiB",
            g.name, g.used_mb, g.total_mb
        );
    }
    if let Some(p) = report.process.as_ref() {
        eprintln!(
            " this process   : {:.0} MiB ({})",
            p.used_mb, p.source
        );
        if let (Some(d), Some(s)) = (p.dedicated_mb, p.shared_mb) {
            eprintln!("   dedicated    : {:.1} MiB   shared {:.1} MiB", d, s);
        }
        if let (Some(free), Some(total)) = (p.cuda_free_mb, p.cuda_total_mb) {
            eprintln!(
                "   cuda device  : {:.0} MiB free / {:.0} MiB total",
                free, total
            );
        }
    } else {
        eprintln!(" this process   : unavailable");
    }
    eprintln!("============================================================");

    let art = std::path::PathBuf::from("artifacts");
    std::fs::create_dir_all(&art).ok();
    let _ = trm_omega::memory::write_vram_report(&art.join("vram_probe.json"), &report);

    let bench = serde_json::json!({
        "variant": format!("{:?}", net_cfg.variant),
        "dim": net_cfg.dim,
        "heads": net_cfg.num_heads,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "n_l": args.n_l,
        "n_sup": args.n_sup,
        "iters": args.iters,
        "secs": secs,
        "iter_per_s": args.iters as f64 / secs,
        "avg_steps_used": steps_sum as f64 / args.iters as f64,
        "packed_linears": model.packed_linear_count(),
        "kernel_backend": format!("{:?}", trm_omega::kernel_dispatch::active_backend()),
        "inference_est_mb": report.inference_est_mb,
        "training_est_mb": report.training_est_mb,
        "readme_inference_mb": report.readme_inference_mb,
        "readme_training_mb": report.readme_training_mb,
        "nvidia_smi": gpu,
        "process": report.process,
        "checksum": checksum,
    });
    let _ = std::fs::write(art.join("kernels_bench.json"), serde_json::to_vec_pretty(&bench)?);

    Ok(())
}
