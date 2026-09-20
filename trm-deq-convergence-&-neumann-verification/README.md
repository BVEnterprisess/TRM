# TRM-Omega

Recursive reasoning engine in Rust + Candle. 2-bit ternary weights, optional CUDA kernels on consumer NVIDIA cards.

Crate: `trm_omega` 10.2. Source lives in this directory (the folder name is historical; the package is `trm_omega`).

## What it is

A small Transformer (or MLP-Mixer) is applied repeatedly over concatenated `[x | y | z]` state:

- **L-cycles / H-cycles** unroll the TRM recursion with optional adaptive halt.
- **DEQ mode** solves the same map to a fixed point with Anderson acceleration, then backprops with a Neumann IFT series instead of an unrolled graph.
- **TRMQ10** packs 2D linears as 2-bit ternary (`00=-1`, `01=0`, `10=+1`) plus per-row alpha.

Default paper-ish shape on this repo: **dim 256, 8 heads, 2 layers, n_L=6, n_sup=16, seq 81**. That is **not** a 7M model. Live `VarMap` count is **2.67M parameters** (embeddings + norms + linears; the default puzzle table is 4096×256). Layer-only estimate is 1.58M. A 2-bit pack of the 2D linears is well under a megabyte.

## Hardware this tree is measured on

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1660, 6 GB, sm_75, WDDM |
| Driver / toolkit | 560.94 / CUDA 12.6 (`nvcc` V12.6.85) |
| Host | Windows 10, MSVC 2022, Rust stable |

Per-process VRAM cannot be read from `nvidia-smi` under WDDM. The binaries sample **VidMm dedicated usage** (same source as Task Manager) and `cuMemGetInfo`.

### Measured (2026-09-20, `--features cuda --release --bin server -- --kernels`)

| Config | Throughput | This process | Board (`nvidia-smi`) |
|---|---|---|---|
| dim 256, seq 81, batch 1, n_L=6, n_sup=16, 2 layers, packed MHA | **0.61 iter/s** (was 0.21 before fused attention) | **97 MiB dedicated** (VidMm) | ~1.3 / 6.1 GiB (desktop share included) |
| VRAM estimate (weights + context + `[x\|y\|z]` maps) | | inf **237 MB** (README band 150–300) | train **~2.4 GB** (README ~2 GB) |

Packed inference: **16 ternary linears**, KernelBank backend **Cuda**. Weights stay on the device after the first call. QKV / RoPE / fused attention / SwiGLU / LayerNorm run in PTX. Activations still cross the host between Candle tensors and KernelBank; that copy is the remaining speed limit, not GEMM.

## Features

| Feature | What it does |
|---|---|
| `cpu` (default) | Candle on CPU. Packed path uses `kernel_ref` (CPU clones of the `.cu` loops). |
| `cuda` | Loads nvcc PTX into `cudarc` KernelBank. Inference / quantized GEMM only. Training autograd stays on Candle. |
| `candle-cuda` | `cuda` plus Candle tensors on GPU (`Device::new_cuda(0)`). Needs the CUDA toolkit at **link** time. Training GEMMs can live on the 1660; packed kernels still round-trip through host buffers today. |

```bash
cargo test --features cpu
cargo test --features cuda --test kernel_parity
cargo run --release --features cuda --bin server -- --kernels --batch 1 --iters 4 --n-l 6 --n-sup 16
```

Windows CUDA builds need `CUDA_PATH` (v12.6 on this rig) and `vcvars64.bat` on PATH.

Unattended toolkit install if you are missing nvcc:

```powershell
.\setup.ps1
# or
cargo run --bin setup -- --install
```

```bash
./setup.sh
```

## Layout

```
src/
  network.rs         Shared f_0: RoPE, GQA, SwiGLU, Mixer, packed MHA
  recursion.rs       TRM L/H cycles, halt head, concat [x|y|z]
  deq.rs             Anderson forward + Neumann IFT backward
  quantize.rs        2-bit ternary pack / STE
  kernel_ref.rs      CPU clones of src/kernels/*.cu
  kernel_dispatch.rs CUDA KernelBank or kernel_ref
  custom_kernels.rs  cudarc loader, device pack cache, fused hosts
  kernels/*.cu       omega, activations, attention, mixer, recurse
  memory.rs          VRAM estimate + WDDM / CUDA sampling
  trmq10.rs          .trmq10 bundle
  train/             AdamW, EMA, loss, trainer
  bin/train.rs eval.rs forge.rs server.rs setup.rs
```

Training and inference are the same `TrmModel`. `--kernels` / `pack_ternary_inference` is inference-only (no grad through packed weights).

## Train / eval / forge / serve

```bash
cargo run --release --bin train -- --task arc --data-dir ./data --dim 256 --heads 8 --layers 2 --l-cycles 6 --n-sup 16
cargo run --release --bin eval -- --model checkpoints/best.safetensors --data-dir ./data/arc
cargo run --release --bin forge -- --input checkpoints/best.safetensors --output model.trmq10
cargo run --release --features cuda --bin server -- --model model.trmq10 --kernels --batch 1 --seq-len 81
```

`server` writes `artifacts/kernels_bench.json` and `artifacts/vram_probe.json`.

## Implementation status

**Done**

- TRM recursion, DEQ Anderson + Neumann IFT, STE ternary, TRMQ10
- Trainer, checkpoints, EMA, eval / forge binaries
- CUDA 12.6 KernelBank on sm_75 (ternary GEMM, bias, SwiGLU, LayerNorm, RoPE, fused attention)
- Device-resident packed weights; fused packed FFN and packed MHA
- WDDM per-process VRAM sampling
- CPU CI (`cargo test --features cpu`)

**Next**

1. End-to-end device activations (no host bounce per layer). `candle-cuda` is the training half; KernelBank needs device pointers from Candle CUDA storage for inference.
2. Honest scaling: this default is **2.67M** live params, not 7M. A 7M run needs a different dim/depth.
3. Mixed-precision training that actually fits comfortably in 6 GB with DEQ + Adam.
4. Data loaders / ARC training that is more than the smoke tests.

**Not goals**

- TCC on this 1660 (it is the display GPU)
- Replacing Candle autograd with custom training kernels
- Gemini / AI Studio / Cloud Run (removed)

## License

MIT. See `LICENSE`.
