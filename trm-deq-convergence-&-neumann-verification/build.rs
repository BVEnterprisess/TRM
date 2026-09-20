use anyhow::{Context, Result};
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    if let Err(e) = run() {
        panic!("\n╔══ build.rs FAILED ══╗\n{e:#}\n╚═════════════════════╝\n");
    }
}

fn detect_nvcc() -> Result<PathBuf> {
    // Check standard environment variables
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(p) = env::var(var) {
            let bin = PathBuf::from(&p).join("bin");
            for name in ["nvcc", "nvcc.exe"] {
                let candidate = bin.join(name);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
    }
    // Check standard Linux & Windows file system installations
    for p in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin\\nvcc.exe",
    ] {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Ok(pb);
        }
    }
    // Check path via which (Unix) or where (Windows)
    for lookup_cmd in ["which", "where"] {
        if let Ok(out) = Command::new(lookup_cmd).arg("nvcc").output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                if let Some(first_line) = s.lines().next() {
                    let trimmed = first_line.trim();
                    if !trimmed.is_empty() {
                        let pb = PathBuf::from(trimmed);
                        if pb.exists() {
                            return Ok(pb);
                        }
                    }
                }
            }
        }
    }
    anyhow::bail!("nvcc not found on system")
}

fn detect_compute_caps() -> Vec<String> {
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            let caps: Vec<String> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter_map(|l| {
                    let sm = l.trim().replace('.', "");
                    if sm.len() >= 2 {
                        Some(format!("sm_{sm}"))
                    } else {
                        None
                    }
                })
                .collect();
            if !caps.is_empty() {
                return caps;
            }
        }
    }
    vec!["sm_75".to_string()]
}

const PTX_STUB: &str = "// TRM_PTX_STUB — nvcc was not available at build time.\n.version 7.5\n.target sm_75\n.address_size 64\n";

fn write_ptx_stubs(out_dir: &PathBuf) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    for name in ["omega", "activations", "attention", "mixer", "recurse"] {
        fs::write(out_dir.join(format!("{name}.ptx")), PTX_STUB)?;
    }
    Ok(())
}

fn run() -> Result<()> {
    // Skip CUDA compilation if cpu feature or cuda feature not active or nvcc not found
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        eprintln!("cargo:warning=Building in CPU mode, skipping CUDA kernels");
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").context("OUT_DIR")?);
    // Always emit files so `include_str!` in custom_kernels.rs compiles.
    write_ptx_stubs(&out_dir)?;

    let nvcc = match detect_nvcc() {
        Ok(p) => p,
        Err(_) => {
            // Check if user requested zero-HITL auto-installation via environment flag
            if env::var("TRM_AUTO_INSTALL").unwrap_or_default() == "1" {
                eprintln!("cargo:warning=nvcc not found and TRM_AUTO_INSTALL=1 detected. Running unattended installer...");
                attempt_auto_install_cuda();
                // Retry detection post-install
                if let Ok(p2) = detect_nvcc() {
                    eprintln!("cargo:warning=unattended installation succeeded; nvcc at {}", p2.display());
                    p2
                } else {
                    eprintln!("cargo:warning=installer finished but nvcc was not on PATH yet. Falling back to CPU kernels.");
                    return Ok(());
                }
            } else {
                eprintln!("cargo:warning=[CUDA MISSING] nvcc was not found on your system.");
                eprintln!("cargo:warning=To install automatically without prompts, run:");
                eprintln!("cargo:warning=  cargo run --bin setup -- --install");
                eprintln!("cargo:warning=Or run: ./setup.ps1 (Windows) or ./setup.sh (Linux)");
                eprintln!("cargo:warning=Proceeding with CPU kernel_ref fallback (PTX stubs written).");
                return Ok(());
            }
        }
    };

    let archs = detect_compute_caps();
    let kernels = ["omega", "activations", "attention", "mixer", "recurse"];
    for name in &kernels {
        let cu = PathBuf::from(format!("src/kernels/{name}.cu"));
        println!("cargo:rerun-if-changed={}", cu.display());
        if !cu.exists() {
            continue;
        }
        let ptx_out = out_dir.join(format!("{name}.ptx"));
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-ptx", "-O3", "--use_fast_math", "-lineinfo", "-std=c++14"]);
        let arch = archs.first().cloned().unwrap_or_else(|| "sm_75".to_string());
        let compute = arch.replace("sm_", "compute_");
        cmd.arg(format!("-arch={compute}"));
        cmd.arg("-o").arg(&ptx_out).arg(&cu);
        let status = cmd.status().with_context(|| format!("Failed to run nvcc for {name}.cu"))?;
        if !status.success() {
            anyhow::bail!("nvcc failed for {name}.cu (exit: {status})");
        }
    }
    Ok(())
}

fn attempt_auto_install_cuda() {
    let is_windows = cfg!(windows) || env::consts::OS == "windows";
    if is_windows {
        let _ = Command::new("winget")
            .args([
                "install",
                "--id",
                "Nvidia.CUDA",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--force",
                "--disable-interactivity",
            ])
            .status();
    } else if PathBuf::from("/usr/bin/apt-get").exists() {
        let _ = Command::new("apt-get")
            .env("DEBIAN_FRONTEND", "noninteractive")
            .args(["update", "-qq"])
            .status();
        let _ = Command::new("apt-get")
            .env("DEBIAN_FRONTEND", "noninteractive")
            .args(["install", "-y", "-qq", "nvidia-cuda-toolkit", "build-essential"])
            .status();
    }
}

