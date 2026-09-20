use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Default)]
pub struct DependencyReport {
    pub gpu_detected: bool,
    pub gpu_name: Option<String>,
    pub compute_cap: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_available: bool,
    pub nvcc_path: Option<PathBuf>,
    pub nvcc_version: Option<String>,
    pub host_compiler_available: bool,
    pub host_compiler_info: Option<String>,
    pub rust_version: Option<String>,
    pub os_name: String,
    pub is_windows: bool,
    pub is_linux: bool,
}

impl DependencyReport {
    pub fn is_fully_cuda_ready(&self) -> bool {
        self.gpu_detected && self.cuda_available && self.host_compiler_available
    }
}

/// Detects all toolchains, GPU drivers, and CUDA dependencies.
pub fn check_dependencies() -> DependencyReport {
    let mut report = DependencyReport {
        os_name: env::consts::OS.to_string(),
        is_windows: cfg!(windows) || env::consts::OS == "windows",
        is_linux: cfg!(unix) || env::consts::OS == "linux",
        ..Default::default()
    };

    // 1. Rust version
    if let Ok(out) = Command::new("rustc").arg("--version").output() {
        if out.status.success() {
            report.rust_version = Some(String::from_utf8_lossy(&out.stdout).trim().to_string());
        }
    }

    // 2. NVIDIA GPU & Driver
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=gpu_name,driver_version,compute_cap", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = stdout.lines().next() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    report.gpu_detected = true;
                    report.gpu_name = Some(parts[0].to_string());
                    report.driver_version = Some(parts[1].to_string());
                    let cc = parts[2].replace('.', "");
                    report.compute_cap = Some(format!("sm_{cc}"));
                }
            }
        }
    }

    // 3. NVCC / CUDA Toolkit
    if let Ok(path) = find_nvcc() {
        report.cuda_available = true;
        report.nvcc_path = Some(path.clone());

        if let Ok(out) = Command::new(&path).arg("--version").output() {
            if out.status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                for line in stdout.lines() {
                    if line.contains("release") || line.contains("V") {
                        report.nvcc_version = Some(line.trim().to_string());
                        break;
                    }
                }
            }
        }
    }

    // 4. Host Compiler (cl.exe on Windows, gcc/clang on Linux)
    if report.is_windows {
        if let Ok(out) = Command::new("where").arg("cl").output() {
            if out.status.success() && !out.stdout.is_empty() {
                report.host_compiler_available = true;
                report.host_compiler_info = Some("MSVC cl.exe detected".to_string());
            }
        }
        if !report.host_compiler_available {
            // Check standard VS installation paths
            for vs_path in [
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC",
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC",
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC",
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC",
            ] {
                if Path::new(vs_path).exists() {
                    report.host_compiler_available = true;
                    report.host_compiler_info = Some(format!("MSVC found at {vs_path}"));
                    break;
                }
            }
        }
    } else {
        for cc in ["gcc", "clang", "g++"] {
            if let Ok(out) = Command::new(cc).arg("--version").output() {
                if out.status.success() {
                    report.host_compiler_available = true;
                    if let Some(first) = String::from_utf8_lossy(&out.stdout).lines().next() {
                        report.host_compiler_info = Some(first.trim().to_string());
                    }
                    break;
                }
            }
        }
    }

    report
}

/// Discovers the path to nvcc across Windows and Unix platforms
pub fn find_nvcc() -> Result<PathBuf> {
    // 1. Environment variables
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

    // 2. Direct paths
    for p in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin\\nvcc.exe",
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

    // 3. System PATH lookup via which/where
    for lookup in ["which", "where"] {
        if let Ok(out) = Command::new(lookup).arg("nvcc").output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                if let Some(first_line) = s.lines().next() {
                    let pb = PathBuf::from(first_line.trim());
                    if pb.exists() {
                        return Ok(pb);
                    }
                }
            }
        }
    }

    anyhow::bail!("nvcc not found")
}

/// Pretty-prints a formatted dependency diagnostics report
pub fn print_report(report: &DependencyReport) {
    println!("\n=======================================================");
    println!(" TRM-Omega: System & Hardware Dependency Diagnostics");
    println!("=======================================================");
    println!("Operating System: {}", report.os_name);
    println!("Rust Toolchain:   {}", report.rust_version.as_deref().unwrap_or("NOT FOUND"));

    if report.gpu_detected {
        println!("NVIDIA GPU:       [OK] {}", report.gpu_name.as_deref().unwrap_or("Detected"));
        println!("Driver Version:   {}", report.driver_version.as_deref().unwrap_or("Unknown"));
        println!("Compute Arch:     {}", report.compute_cap.as_deref().unwrap_or("sm_75"));
    } else {
        println!("NVIDIA GPU:       [WARN] Not detected via nvidia-smi (CPU mode will be used)");
    }

    if report.cuda_available {
        println!("CUDA Toolkit:     [OK] {}", report.nvcc_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default());
        if let Some(ref ver) = report.nvcc_version {
            println!("CUDA Version:     {ver}");
        }
    } else {
        println!("CUDA Toolkit:     [MISSING] nvcc was not found on PATH or default locations");
    }

    if report.host_compiler_available {
        println!("C++ Host Tool:    [OK] {}", report.host_compiler_info.as_deref().unwrap_or("Installed"));
    } else {
        println!("C++ Host Tool:    [MISSING] C++ compiler (MSVC cl.exe / gcc) required for CUDA build");
    }

    println!("=======================================================");
    if report.is_fully_cuda_ready() {
        println!("STATUS: ALL DEPENDENCIES SATISFIED FOR FULL HARDWARE ACCELERATION");
    } else if report.gpu_detected && !report.cuda_available {
        println!("STATUS: NVIDIA GPU PRESENT BUT CUDA TOOLKIT IS MISSING.");
        println!("        Automated zero-HITL installer can install it automatically.");
    } else {
        println!("STATUS: RUNNING IN CPU FALLBACK MODE.");
    }
    println!("=======================================================\n");
}

/// Executes a fully automated, zero-HITL silent installation of CUDA Toolkit and C++ build tools.
pub fn auto_install_cuda(silent: bool) -> Result<()> {
    let report = check_dependencies();

    if report.cuda_available {
        if !silent {
            println!("[auto_install] CUDA Toolkit is already installed: {:?}", report.nvcc_path);
        }
        return Ok(());
    }

    println!("\n>>> STARTING ZERO-HITL AUTOMATED CUDA INSTALLATION <<<");

    if report.is_windows {
        install_cuda_windows(silent)?;
    } else if report.is_linux {
        install_cuda_linux(silent)?;
    } else {
        anyhow::bail!("Unsupported platform for automated zero-HITL installation: {}", report.os_name);
    }

    // Refresh and verify
    let refreshed = check_dependencies();
    if refreshed.cuda_available {
        println!(">>> ZERO-HITL CUDA INSTALLATION COMPLETED SUCCESSFULLY! <<<");
        if let Some(p) = refreshed.nvcc_path {
            println!("Installed at: {}", p.display());
        }
        Ok(())
    } else {
        println!(">>> Installation finished. Note: A shell restart or setting CUDA_PATH may be required if system environment variables were updated. <<<");
        Ok(())
    }
}

fn install_cuda_windows(silent: bool) -> Result<()> {
    if !silent {
        println!("[Windows] Attempting zero-HITL installation via winget...");
    }

    // 1. Try winget (built into Windows 10/11)
    let winget_res = Command::new("winget")
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

    if let Ok(status) = winget_res {
        if status.success() {
            if !silent {
                println!("[Windows] winget installed Nvidia.CUDA successfully.");
            }
            return Ok(());
        }
    }

    // 2. Fallback: Download official NVIDIA CUDA installer directly and run silently
    if !silent {
        println!("[Windows] Winget returned non-zero or was unavailable. Falling back to direct silent installer download...");
    }

    let temp_dir = env::temp_dir();
    let installer_path = temp_dir.join("cuda_installer_silent.exe");

    let download_cmd = format!(
        "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/12.4.1/network_installers/cuda_12.4.1_windows_network.exe' -OutFile '{}'",
        installer_path.display()
    );

    let ps_status = Command::new("powershell")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &download_cmd])
        .status()
        .context("Failed to run PowerShell download")?;

    if ps_status.success() && installer_path.exists() {
        if !silent {
            println!("[Windows] Executing silent installer (-s nvcc_12.4)...");
        }
        let install_status = Command::new(&installer_path)
            .args(["-s", "nvcc_12.4", "cudart_12.4", "visual_studio_integration_12.4"])
            .status()
            .context("Failed to execute silent CUDA installer")?;

        if install_status.success() {
            if !silent {
                println!("[Windows] Silent installer finished successfully.");
            }
        }
    }

    Ok(())
}

fn install_cuda_linux(silent: bool) -> Result<()> {
    if !silent {
        println!("[Linux] Attempting zero-HITL package manager installation...");
    }

    // Check for apt-get (Debian / Ubuntu)
    if Path::new("/usr/bin/apt-get").exists() {
        if !silent {
            println!("[Linux] Detected apt-get. Installing nvidia-cuda-toolkit & build-essential silently...");
        }
        let status = Command::new("apt-get")
            .env("DEBIAN_FRONTEND", "noninteractive")
            .args(["update", "-qq"])
            .status();
        let _ = status;

        let install = Command::new("apt-get")
            .env("DEBIAN_FRONTEND", "noninteractive")
            .args(["install", "-y", "-qq", "nvidia-cuda-toolkit", "build-essential"])
            .status();

        if let Ok(s) = install {
            if s.success() {
                return Ok(());
            }
        }
    }

    // Check for dnf (Fedora / RHEL)
    if Path::new("/usr/bin/dnf").exists() {
        if !silent {
            println!("[Linux] Detected dnf. Installing cuda-toolkit silently...");
        }
        let install = Command::new("dnf")
            .args(["install", "-y", "-q", "cuda-toolkit", "gcc-c++"])
            .status();
        if let Ok(s) = install {
            if s.success() {
                return Ok(());
            }
        }
    }

    // Check for pacman (Arch Linux)
    if Path::new("/usr/bin/pacman").exists() {
        let install = Command::new("pacman")
            .args(["-Sy", "--noconfirm", "--needed", "cuda", "base-devel"])
            .status();
        if let Ok(s) = install {
            if s.success() {
                return Ok(());
            }
        }
    }

    Ok(())
}

/// Checks dependencies and optionally executes automated zero-HITL installation
pub fn ensure_dependencies(auto_install: bool) -> Result<DependencyReport> {
    let report = check_dependencies();
    if !report.cuda_available && report.gpu_detected {
        if auto_install {
            println!("[ensure_dependencies] NVIDIA GPU detected but CUDA Toolkit missing. Triggering zero-HITL auto-installation...");
            let _ = auto_install_cuda(false);
            return Ok(check_dependencies());
        } else {
            log::warn!("NVIDIA GPU detected but CUDA Toolkit is missing. Run with `--auto-install` or execute `./setup.ps1` for zero-HITL setup.");
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_check_runs_and_populates_os() {
        let report = check_dependencies();
        assert!(!report.os_name.is_empty(), "OS name should be detected");
        assert!(report.is_linux || report.is_windows, "Target platform should be identified");
        assert!(report.rust_version.is_some(), "Rust toolchain should be detected");
    }

    #[test]
    fn test_print_report_does_not_panic() {
        let report = check_dependencies();
        print_report(&report);
    }

    #[test]
    fn test_dependency_report_readiness_logic() {
        let mut report = DependencyReport::default();
        assert!(!report.is_fully_cuda_ready());

        report.gpu_detected = true;
        report.cuda_available = true;
        report.host_compiler_available = true;
        assert!(report.is_fully_cuda_ready());

        report.cuda_available = false;
        assert!(!report.is_fully_cuda_ready());
    }
}

