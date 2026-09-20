---
name: observing-gpu-host-bounce
description: Use when profiling TRM-Omega CUDA packed inference, KernelBank HtoD/DtoH, nvidia-smi process VRAM N/A on WDDM, nsys, ncu, compute-sanitizer, or claiming activations stay on the GPU.
---

# Observing GPU host bounce

## Overview

Wall-clock `iter/s` cannot prove copies disappeared. Count KernelBank HtoD/DtoH first; open Nsight only when those counters cannot explain the time.

## Ladder (cheapest first)

| Rung | Tool | Use when |
|------|------|----------|
| 0 | `server --kernels` wall clock + checksum | Always |
| 1 | `copy_stats` in `artifacts/kernels_bench.json` | Always on packed CUDA |
| 2 | WDDM VidMm + `cuMemGetInfo` | VRAM / 1660 fit |
| 3 | `nsys` CUDA API + GPU kernel trace | Counters do not explain wall time, or claiming host bounce is gone |
| 4 | `ncu` on **one** named kernel | nsys shows that kernel is the hotspot |
| 5 | `compute-sanitizer --tool memcheck` | New `.cu` / PTX, tiny test only |

Do not run 3–5 on every compile. Do not run 4 or 5 at paper `n_l=6 n_sup=16`.

## KernelBank counters

`server --kernels` resets copy stats after warmup, then records the timed loop.

**Host bounce still present:** HtoD/DtoH calls per iter stay in the hundreds (LN + MHA + FFN per layer × L × H).

**Activations resident:** about **1 DtoH per `SharedNetwork::forward`**, times L×H steps (paper: **112 DtoH/iter** at n_L=6, n_sup=16). Remaining HtoD is LN / RoPE / mask uploads, not activation bounce. Weight cache uploads belong in warmup, not the timed window.

Read `copy_calls_per_iter` in `artifacts/kernels_bench.json`. nvidia-smi process memory stays `[N/A]` on this WDDM 1660 — that is not a permissions bug.

## Commands

Repo script (finds `nsys`/`ncu`/`compute-sanitizer` on this box):

```powershell
powershell -File .cursor/skills/observing-gpu-host-bounce/scripts/profile-kernels.ps1 -Mode nsys
powershell -File .cursor/skills/observing-gpu-host-bounce/scripts/profile-kernels.ps1 -Mode ncu -Kernel fused_attention
powershell -File .cursor/skills/observing-gpu-host-bounce/scripts/profile-kernels.ps1 -Mode sanitizer
```

Prefer **Nsight Systems** `nsys.exe` (2024.5+). The copy bundled under Nsight Compute 2024.3 fails without admin ETW (`ReflexStatsTraceLoggingProvider`). Profile `server.exe`, not `cargo`.

Install CUDA **12.6** Nsight pieces only (never `winget Nvidia.CUDA` 13.x over this toolkit):

```powershell
powershell -File .cursor/skills/observing-gpu-host-bounce/scripts/install-nsight.ps1
```

## Common mistakes

| Excuse | Reality |
|--------|---------|
| "iter/s went up, bounce is gone" | Need `copy_calls_per_iter` drop, then nsys CUDA memcpy vs kernel time |
| "ncu the whole server" | ncu one kernel after nsys names it |
| "sanitizer on paper config" | Use `kernel_parity` / tiny `--iters 1 --n-l 1 --n-sup 1` |
| "nvidia-smi process VRAM" | Use VidMm + `cuda_free_mb` |
| "install CUDA 13 for nsys" | Add `nsight_systems_12.6` to the existing 12.6 toolkit |

## Red flags

- About to tile `fused_attention` without an nsys hotspot on that kernel
- About to claim device-resident activations without `copy_stats`
- Running nsys as a default post-build step
