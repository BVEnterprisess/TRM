use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use trm_omega::default_device;
use trm_omega::network::{NetworkConfig, NetworkVariant};
use trm_omega::preset;
use trm_omega::recursion::{TrmConfig, TrmModel};
use trm_omega::train::trainer::{TaskType, TrainConfig, Trainer};

#[derive(Parser, Debug)]
#[command(about = "Train TRM-Omega v10.2")]
struct Args {
    #[arg(long, default_value = "arc")]
    task: String,

    #[arg(long, default_value = "data")]
    data_dir: String,

    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,

    #[arg(long)]
    resume: Option<String>,

    /// Named architecture: tiny (dim 32), paper (dim 256 / 2.67M), 7m (dim 448 / ~7M).
    /// Overrides --dim / --heads / --layers.
    #[arg(long)]
    preset: Option<String>,

    #[arg(long, default_value_t = 256)]
    dim: usize,

    #[arg(long, default_value_t = 8)]
    heads: usize,

    #[arg(long, default_value_t = 2)]
    layers: usize,

    #[arg(long, default_value_t = 11)]
    vocab: usize,

    #[arg(long, default_value_t = 6)]
    l_cycles: usize,

    #[arg(long, default_value_t = 16)]
    n_sup: usize,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    #[arg(long, default_value_t = 4)]
    micro_batch: usize,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    fp16: bool,

    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    #[arg(long, default_value_t = 1.0)]
    wd: f64,

    #[arg(long, default_value_t = 50000)]
    epochs: usize,

    /// Stop after this many optimizer steps (cloud 1-step VRAM soaks).
    #[arg(long)]
    max_steps: Option<usize>,

    #[arg(long, default_value_t = 5000)]
    eval_every: usize,

    #[arg(long, default_value = "transformer")]
    variant: String,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = false)]
    no_augment: bool,

    #[arg(long, default_value_t = false)]
    deq: bool,

    #[arg(long, default_value_t = false)]
    auto_install: bool,

    /// Build the model, print live param count, exit. No data, no GPU required.
    #[arg(long, default_value_t = false)]
    print_params: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let (dim, heads, layers) = preset::resolve(args.preset.as_deref(), args.dim, args.heads, args.layers)?;
    if let Some(name) = args.preset.as_deref() {
        eprintln!("preset '{name}' -> dim={dim} heads={heads} layers={layers}");
    }

    let task_type = match args.task.as_str() {
        "sudoku" => TaskType::Sudoku,
        "maze" => TaskType::Maze,
        _ => TaskType::Arc,
    };

    let variant = match args.variant.as_str() {
        "mixer" => NetworkVariant::MlpMixer,
        _ => NetworkVariant::Transformer,
    };

    let max_seq = match task_type {
        TaskType::Sudoku => 81,
        _ => 900,
    };

    let net_cfg = NetworkConfig {
        variant,
        dim,
        num_heads: heads,
        num_layers: layers,
        vocab_size: args.vocab,
        max_seq_len: max_seq * 3,
        ..Default::default()
    };

    let trm_cfg = TrmConfig {
        n_l_cycles: args.l_cycles,
        n_sup: args.n_sup,
        collect_intermediates: true,
        use_learned_halt_head: true,
        use_deq: args.deq,
        z_seq: 81,
        ..Default::default()
    };

    if args.print_params {
        let model = TrmModel::new(candle_core::Device::Cpu, net_cfg, trm_cfg)?;
        let n = model.param_count();
        eprintln!(
            "live_params={} ({:.3}M)  dim={} heads={} layers={} vocab={} max_seq={} z_seq={} puzzles={}",
            n,
            n as f64 / 1e6,
            model.net_cfg.dim,
            model.net_cfg.num_heads,
            model.net_cfg.num_layers,
            model.net_cfg.vocab_size,
            model.net_cfg.max_seq_len,
            model.trm_cfg.z_seq,
            model.trm_cfg.max_puzzles,
        );
        return Ok(());
    }

    let _ = trm_omega::setup::ensure_dependencies(args.auto_install);

    let device = default_device()?;

    let train_cfg = TrainConfig {
        data_dir: PathBuf::from(&args.data_dir),
        task_type,
        batch_size: args.batch_size,
        micro_batch_size: args.micro_batch,
        use_fp16: args.fp16,
        base_lr: args.lr,
        weight_decay: args.wd,
        max_epochs: args.epochs,
        eval_every_epochs: args.eval_every,
        checkpoint_dir: PathBuf::from(&args.checkpoint_dir),
        resume_tag: args.resume.clone(),
        seed: args.seed,
        no_augment: args.no_augment,
        max_steps: args.max_steps,
        ..Default::default()
    };

    let mut trainer = Trainer::new(device, net_cfg, trm_cfg, train_cfg)?;
    if args.resume.is_some() {
        trainer.resume()?;
    }

    trainer.train()
}
