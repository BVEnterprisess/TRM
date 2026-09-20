# TRM-Omega

Recursive reasoning engine in Rust + [Candle](https://github.com/huggingface/candle). 2-bit ternary weights, optional CUDA kernels, measured on a 6 GB GTX 1660.

Package: `trm_omega` 10.2. Source: [`trm_omega/`](./trm_omega/).

## Quick start

```bash
cd trm_omega
cargo test --features cpu
```

CUDA packed inference (needs CUDA 12.6 + `nvcc` on PATH):

```bash
cargo test --features cuda --test kernel_parity
cargo run --release --features cuda --bin server -- --kernels --batch 1 --iters 4 --n-l 6 --n-sup 16
```

Train **off-box** on a rented Linux GPU, then forge and infer on the 1660:

```bash
# Rented box (candle-cuda does not compile KernelBank)
bash scripts/cloud_train.sh                 # STAGE=print: instantiate 7m, print live params
STAGE=smoke bash scripts/cloud_train.sh     # dim 32, 1 step, seq 81
STAGE=7m    bash scripts/cloud_train.sh     # dim 448 ~7M, 1-step soak
STAGE=train bash scripts/cloud_train.sh     # real run
STAGE=forge bash scripts/cloud_train.sh

# 1660: copy the .trmq10 (or .safetensors), build server locally
cargo run --release --features cuda --bin server -- --model model.trmq10 --kernels --preset 7m --seq-len 81 --batch 1 --iters 4
```

Details, measured numbers, and layout live in [`trm_omega/README.md`](./trm_omega/README.md).

## What it is

A small Transformer (or MLP-Mixer) is applied repeatedly over concatenated `[x | y | z]` state (question, answer track, latent). Unrolled L/H cycles with optional halt, or DEQ Anderson + Neumann IFT. Inference packs 2D linears as 2-bit ternary (TRMQ10).

Default paper-ish shape is **dim 256, 8 heads, 2 layers**. Live count is **2.67M**. `--preset 7m` is dim 448 / 8 heads / 2 layers (**6,787,969** / 6.788M). Train that on a rented GPU; the 1660 is inference after 2-bit forge.

## License

MIT. See [`LICENSE`](./LICENSE).
