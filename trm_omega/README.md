# TRM-Omega

Recursive reasoning engine in Rust + Candle. 2-bit ternary weights, optional CUDA kernels on consumer NVIDIA cards.

Crate: `trm_omega` 10.2.

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

### Packed inference (2026-09-20, `--features cuda --release --bin server -- --kernels`)

Device-resident packed transformer stack: one activation HtoD into the network, LayerNorm / MHA / SwiGLU / residual on device, one DtoH out. Remaining HtoD is LN / RoPE / mask uploads, not per-layer activation bounce.

| Config | Throughput | Copies / iter | This process | Board (`nvidia-smi`) |
|---|---|---|---|---|
| dim 256, seq 81, batch 1, n_L=6, n_sup=16, 2 layers | **1.06 iter/s** (0.21 → 0.61 fused attention → 0.71 device stack → 1.06 tiled attention) | 1344 HtoD / **112 DtoH** | **97 MiB dedicated** (VidMm) | ~1.3 / 6.1 GiB (desktop share included) |
| VRAM estimate | | | inf **237 MB** (README band 150–300) | train **~5.0 GB measured** (was estimated ~2.4 GB) |

Packed inference: **16 ternary linears**, KernelBank backend **Cuda**. Weights stay on the device after the first call. QKV / RoPE / fused attention / SwiGLU / LayerNorm run in PTX.

nsys (n_L=2, n_sup=2, Nsight Systems 2024.5): naive `fused_attention` was **68.4%** of GPU kernel time (avg 8.0 ms). After cooperative QK + K/V tiles + online softmax: **52.0%** (avg 3.8 ms). `ternary_matmul_stack` is now **45.5%**. HtoD/DtoH API time stays under 1%. Paper packed wall clock after the tile: **1.06 iter/s**.

### Product loop (2026-09-20)

Tiny seed-ARC pass to prove train → forge → kernels → eval on this GPU (not a quality run):

| Stage | Result |
|---|---|
| `train` `--features candle-cuda` dim 32, 1 epoch, 5 train / 1 val | Device::new_cuda, 0.16M live params, loss 29.0882, `artifacts/loop_ckpt/final.safetensors` |
| `forge` | `artifacts/loop.trmq10` (19 ternary, 28 fp32) |
| `server --kernels` dim 32, n_L=1, n_sup=2, 2 iters | 0.86 iter/s, checksum 640, 1344/112 copies per iter, 97 MiB dedicated |
| `eval` 4 seed puzzles, 1 augmentation | 0/4 exact (expected at 1 epoch) |

Paper-config training **fits** on this 1660 but is tight: **5038 MiB dedicated** (VidMm), board **5873 / 6144 MiB**, 1 sudoku step in 10.1s (`artifacts/paper_ckpt/train_vram.json`). The old ~2.4 GB estimate undercounted Candle's autograd graph. `--fp16` is still a no-op (tensors stay F32). Do not treat 0% exact as a model bug.

## Features

| Feature | What it does |
|---|---|
| `cpu` (default) | Candle on CPU. Packed path uses `kernel_ref` (CPU clones of the `.cu` loops). |
| `cuda` | Loads nvcc PTX into `cudarc` KernelBank. Inference / quantized GEMM only. Training autograd stays on Candle. |
| `candle-cuda` | `cuda` plus Candle tensors on GPU (`Device::new_cuda(0)`). Needs the CUDA toolkit at **link** time. Training GEMMs can live on the 1660. Packed inference still uses KernelBank, not Candle CUDA storage pointers. |

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

`--fp16` defaults on; disable with `--fp16=false` (not `--fp16 false`).

`server` writes `artifacts/kernels_bench.json` and `artifacts/vram_probe.json`.

## Implementation status

**Done**

- TRM recursion, DEQ Anderson + Neumann IFT, STE ternary, TRMQ10
- Trainer, checkpoints, EMA, eval / forge binaries
- CUDA 12.6 KernelBank on sm_75 (ternary GEMM, bias, SwiGLU, LayerNorm, RoPE, tiled fused attention)
- Device-resident packed weights and packed transformer activations (one DtoH per network forward)
- WDDM per-process VRAM sampling; live `param_count` in the VRAM estimator
- `candle-cuda` train on this 1660: tiny dim-32 seed ARC, and paper-config sudoku (dim 256, n_L=6, n_sup=16, seq 81) at **5.0 GB dedicated**
- Full product pass: train → forge TRMQ10 → `server --kernels` → eval
- CPU CI (`cargo test --features cpu`)

**Next**

1. `ternary_matmul_stack` is co-hotspot with attention on packed inference (nsys ~45%). Speed work, not a close-out blocker.
2. Wire `--fp16` or DEQ if you want headroom under 6 GB; F32 paper train leaves ~270 MiB free on the board.
3. Honest 7M scaling: this default is **2.67M** live params. A 7M run needs a different dim/depth and likely will not fit this card in F32 unroll.
4. Data loaders / ARC training that is more than the smoke tests (quality, not the 1660 stack).

**Not goals**

- TCC on this 1660 (it is the display GPU)
- Replacing Candle autograd with custom training kernels
- Gemini / AI Studio / Cloud Run (removed)

## License

MIT. See `LICENSE`.
