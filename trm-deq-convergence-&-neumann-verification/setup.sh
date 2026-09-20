#!/usr/bin/env bash
# TRM-Omega unattended setup for Linux / WSL
set -e

echo -e "\n\033[1;36m=======================================================\033[0m"
echo -e "\033[1;36m TRM-Omega: unattended environment setup (Linux/WSL)\033[0m"
echo -e "\033[1;36m=======================================================\033[0m"

# 1. Detect GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=gpu_name,driver_version --format=csv,noheader,nounits | head -n 1)
    echo -e "\033[1;32m[OK] GPU Detected:\033[0m $GPU_INFO"
else
    echo -e "\033[1;33m[WARN] nvidia-smi not found. Running on CPU or inside non-passthrough container.\033[0m"
fi

# 2. Check Rust
if command -v cargo &>/dev/null; then
    echo -e "\033[1;32m[OK] Rust toolchain detected:\033[0m $(rustc --version)"
else
    echo -e "\033[1;33m[INFO] Installing Rust toolchain silently...\033[0m"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi

# 3. Check C++ Compiler
if command -v gcc &>/dev/null || command -v clang &>/dev/null; then
    echo -e "\033[1;32m[OK] Host C++ compiler detected.\033[0m"
else
    echo -e "\033[1;33m[INFO] Installing build essentials...\033[0m"
    if command -v apt-get &>/dev/null; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && sudo apt-get install -y -qq build-essential
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y -q gcc-c++ make
    elif command -v pacman &>/dev/null; then
        sudo pacman -Sy --noconfirm base-devel
    fi
fi

# 4. Check CUDA Toolkit (nvcc)
if command -v nvcc &>/dev/null || [ -f "/usr/local/cuda/bin/nvcc" ]; then
    echo -e "\033[1;32m[OK] CUDA Toolkit detected.\033[0m"
else
    echo -e "\033[1;33m[INFO] CUDA Toolkit missing. Initiating zero-HITL silent installation...\033[0m"
    if command -v apt-get &>/dev/null; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-cuda-toolkit
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y -q cuda-toolkit
    elif command -v pacman &>/dev/null; then
        sudo pacman -Sy --noconfirm cuda
    fi
fi

echo -e "\033[1;36m=======================================================\033[0m"
echo -e "\033[1;32m Setup complete.\033[0m"
echo -e "\033[1;36m You can now build and run:\033[0m"
echo -e "   cargo run --release --bin train -- --task arc"
echo -e "   cargo run --release --bin server"
echo -e "\033[1;36m=======================================================\n\033[0m"
