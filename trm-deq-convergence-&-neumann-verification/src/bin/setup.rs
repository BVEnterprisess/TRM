use anyhow::Result;
use clap::Parser;
use trm_omega::setup::{auto_install_cuda, check_dependencies, print_report};

#[derive(Parser, Debug)]
#[command(name = "setup", about = "TRM-Omega dependency and hardware setup")]
struct Args {
    /// Perform full diagnostics check
    #[arg(long, default_value_t = true)]
    check: bool,

    /// Automatically install missing dependencies (CUDA Toolkit, build tools) without user prompts
    #[arg(long, default_value_t = false)]
    install: bool,

    /// Run installation silently
    #[arg(long, default_value_t = false)]
    silent: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let report = check_dependencies();
    if !args.silent {
        print_report(&report);
    }

    if args.install || (!report.cuda_available && report.gpu_detected) {
        if args.install {
            println!("Installing missing CUDA / build-tool dependencies...");
            auto_install_cuda(args.silent)?;
            let final_report = check_dependencies();
            if !args.silent {
                print_report(&final_report);
            }
        } else {
            println!("Note: To trigger automatic unattended installation of missing components, pass `--install`:");
            println!("  cargo run --bin setup -- --install\n");
        }
    }

    Ok(())
}
