# TRM-Omega unattended setup for Windows
# NVIDIA Turing / GTX 1660 / sm_75
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " TRM-Omega: unattended environment setup" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Check NVIDIA GPU
$gpuName = ""
$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuOut = & nvidia-smi --query-gpu=gpu_name,driver_version --format=csv,noheader,nounits 2>$null
    if ($gpuOut) {
        $gpuName = $gpuOut[0].Trim()
        Write-Host "[OK] Detected GPU: $gpuName" -ForegroundColor Green
    }
} else {
    Write-Host "[WARN] nvidia-smi not found. GPU driver may need installation." -ForegroundColor Yellow
}

# 2. Check Rust Toolchain
$cargo = Get-Command "cargo" -ErrorAction SilentlyContinue
if (-not $cargo) {
    Write-Host "[INFO] Rust toolchain missing. Installing Rustup silently..." -ForegroundColor Yellow
    $rustupInstaller = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInstaller
    Start-Process -FilePath $rustupInstaller -ArgumentList "-y", "--default-toolchain", "stable" -Wait -NoNewWindow
    $env:PATH += ";$env:USERPROFILE\.cargo\bin"
    Write-Host "[OK] Rust toolchain installed." -ForegroundColor Green
} else {
    Write-Host "[OK] Rust toolchain detected." -ForegroundColor Green
}

# 3. Check MSVC C++ Build Tools
$cl = Get-Command "cl" -ErrorAction SilentlyContinue
$vsFound = $false
if ($cl) {
    $vsFound = $true
} else {
    $vsCandidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC"
    )
    foreach ($cand in $vsCandidates) {
        if (Test-Path $cand) {
            $vsFound = $true
            break
        }
    }
}

if (-not $vsFound) {
    Write-Host "[INFO] MSVC C++ compiler missing. Installing Visual Studio Build Tools silently..." -ForegroundColor Yellow
    $winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if ($winget) {
        & winget install --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-package-agreements --accept-source-agreements --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"
    }
} else {
    Write-Host "[OK] MSVC C++ Build Tools detected." -ForegroundColor Green
}

# 4. Check CUDA Toolkit (nvcc)
$nvccPath = ""
$nvccCmd = Get-Command "nvcc" -ErrorAction SilentlyContinue
if ($nvccCmd) {
    $nvccPath = $nvccCmd.Source
} else {
    $cudaCandidates = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
    )
    foreach ($cand in $cudaCandidates) {
        if (Test-Path $cand) {
            $nvccPath = $cand
            $cudaBin = Split-Path $cand -Parent
            $cudaRoot = Split-Path $cudaBin -Parent
            $env:CUDA_PATH = $cudaRoot
            $env:PATH = "$cudaBin;" + $env:PATH
            [Environment]::SetEnvironmentVariable("CUDA_PATH", $cudaRoot, "User")
            break
        }
    }
}

if (-not $nvccPath) {
    Write-Host "[INFO] CUDA Toolkit missing. Initiating zero-HITL silent installation..." -ForegroundColor Yellow
    $winget = Get-Command "winget" -ErrorAction SilentlyContinue
    $installed = $false
    if ($winget) {
        Write-Host "[INFO] Running winget install Nvidia.CUDA silently..." -ForegroundColor Cyan
        & winget install --id Nvidia.CUDA --silent --accept-package-agreements --accept-source-agreements --force --disable-interactivity
        
        # Check standard install location post-winget
        $postCandidates = @(
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"
        )
        foreach ($c in $postCandidates) {
            if (Test-Path $c) {
                $nvccPath = $c
                $cudaBin = Split-Path $c -Parent
                $cudaRoot = Split-Path $cudaBin -Parent
                $env:CUDA_PATH = $cudaRoot
                $env:PATH = "$cudaBin;" + $env:PATH
                $installed = $true
                break
            }
        }
    }

    if (-not $installed) {
        Write-Host "[INFO] Downloading official CUDA network installer for silent setup..." -ForegroundColor Cyan
        $cudaInstaller = "$env:TEMP\cuda_setup_network.exe"
        Invoke-WebRequest -Uri "https://developer.download.nvidia.com/compute/cuda/12.4.1/network_installers/cuda_12.4.1_windows_network.exe" -OutFile $cudaInstaller
        Start-Process -FilePath $cudaInstaller -ArgumentList "-s", "nvcc_12.4", "cudart_12.4" -Wait -NoNewWindow
    }
} else {
    Write-Host "[OK] CUDA Toolkit detected at: $nvccPath" -ForegroundColor Green
}

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Setup complete." -ForegroundColor Green
Write-Host " You can now run:" -ForegroundColor Cyan
Write-Host "   cargo run --release --bin train -- --task arc" -ForegroundColor White
Write-Host "   cargo run --release --bin server" -ForegroundColor White
Write-Host "=======================================================`n" -ForegroundColor Cyan
