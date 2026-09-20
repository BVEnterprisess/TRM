use std::path::PathBuf;
use anyhow::Result;
use clap::Parser;
use trm_omega::data::load_arc_tasks;
use trm_omega::default_device;
use trm_omega::network::{NetworkConfig, NetworkVariant};
use trm_omega::recursion::TrmConfig;
use trm_omega::task_eval::evaluate_arc_tasks;
use trm_omega::trmq10;

#[derive(Parser, Debug)]
#[command(about = "Evaluate TRM-Omega on ARC-like tasks")]
struct Args {
    #[arg(long)] model: String,
    #[arg(long)] data_dir: PathBuf,
    #[arg(long, default_value_t = 1000)] n_augmentations: usize,
    #[arg(long, default_value_t = 256)] dim: usize,
    #[arg(long, default_value_t = 8)] heads: usize,
    #[arg(long, default_value_t = 11)] vocab: usize,
    #[arg(long, default_value_t = 2)] layers: usize,
    #[arg(long, default_value_t = true)] two_try: bool,
    #[arg(long, default_value_t = false)] deq: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = default_device()?;
    let net_cfg = NetworkConfig {
        variant: NetworkVariant::Transformer,
        dim: args.dim,
        num_heads: args.heads,
        max_seq_len: 900 * 3,
        vocab_size: args.vocab,
        num_layers: args.layers,
        ..Default::default()
    };
    let trm_cfg = TrmConfig {
        use_deq: args.deq,
        ..Default::default()
    };
    let model = trmq10::load_model(device.clone(), &args.model, net_cfg, trm_cfg)?;
    let tasks = load_arc_tasks(&args.data_dir)?;
    let result = evaluate_arc_tasks(
        &model,
        &device,
        &tasks,
        args.n_augmentations,
        args.two_try,
    )?;
    eprintln!("\n total test inputs : {}", result.total);
    eprintln!(" exact matches    : {}", result.exact);
    eprintln!(" exact accuracy   : {:.2}%", 100.0 * result.accuracy());
    Ok(())
}
