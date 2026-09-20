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

Train on Candle CUDA, forge a TRMQ10 bundle, then eval:

```bash
cargo run --release --features candle-cuda --bin train -- --task arc --data-dir artifacts/loop_data --fp16=false --epochs 1 --dim 32 --heads 4 --layers 2
cargo run --release --features cuda --bin forge -- --input artifacts/loop_ckpt/final.safetensors --output artifacts/loop.trmq10 --dim 32 --heads 4 --layers 2
cargo run --release --features cuda --bin eval -- --model artifacts/loop.trmq10 --data-dir artifacts/loop_data --dim 32 --heads 4 --layers 2
```

Details, measured numbers, and layout live in [`trm_omega/README.md`](./trm_omega/README.md).

## What it is

A small Transformer (or MLP-Mixer) is applied repeatedly over concatenated `[x | y | z]` state (question, answer track, latent). Unrolled L/H cycles with optional halt, or DEQ Anderson + Neumann IFT. Inference packs 2D linears as 2-bit ternary (TRMQ10).

Default paper-ish shape is **dim 256, 8 heads, 2 layers, n_L=6, n_sup=16**. Live parameter count is **2.67M**, not 7M.

## License

MIT. See [`LICENSE`](./LICENSE).
