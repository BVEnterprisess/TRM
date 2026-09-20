use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use trm_omega::default_device;
use trm_omega::network::{NetworkConfig, NetworkVariant};
use trm_omega::recursion::TrmConfig;
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
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Check system hardware dependencies and optionally auto-install missing tools
    let _ = trm_omega::setup::ensure_dependencies(args.auto_install);

    let device = default_device()?;

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
        dim: args.dim,
        num_heads: args.heads,
        num_layers: args.layers,
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
        ..Default::default()
    };

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
        ..Default::default()
    };

    let mut trainer = Trainer::new(device, net_cfg, trm_cfg, train_cfg)?;
    if args.resume.is_some() {
        trainer.resume()?;
    }

    trainer.train()
}
