use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use trm_omega::default_device;
use trm_omega::network::NetworkConfig;
use trm_omega::preset;
use trm_omega::recursion::{TrmConfig, TrmModel};
use trm_omega::trmq10;

#[derive(Parser, Debug)]
#[command(about = "Forge a quantized deployment bundle from candle safetensors")]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    /// Named architecture: tiny, paper, 7m. Overrides --dim / --heads / --layers.
    #[arg(long)]
    preset: Option<String>,

    #[arg(long, default_value_t = 256)]
    dim: usize,

    #[arg(long, default_value_t = 8)]
    heads: usize,

    #[arg(long, default_value_t = 900)]
    max_seq: usize,

    #[arg(long, default_value_t = 11)]
    vocab: usize,

    #[arg(long, default_value_t = 2)]
    layers: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let (dim, heads, layers) = preset::resolve(args.preset.as_deref(), args.dim, args.heads, args.layers)?;
    if let Some(name) = args.preset.as_deref() {
        eprintln!("preset '{name}' -> dim={dim} heads={heads} layers={layers}");
    }

    let device = default_device()?;
    let net_cfg = NetworkConfig {
        dim,
        num_heads: heads,
        max_seq_len: args.max_seq,
        vocab_size: args.vocab,
        num_layers: layers,
        ..Default::default()
    };
    let trm_cfg = TrmConfig::default();
    let model = TrmModel::load_safetensors(device, net_cfg, trm_cfg, &args.input)?;
    trmq10::save_trmq10(&args.output, &model)?;
    eprintln!("forged {}", args.output.display());
    Ok(())
}
