# TRM

Rust workspace for **TRM-Omega**: a recursive reasoning engine (Candle + optional CUDA kernels) aimed at a 6 GB GTX 1660.

The crate is in [`trm-deq-convergence-&-neumann-verification/`](./trm-deq-convergence-&-neumann-verification/) (historical AI Studio folder name; package is `trm_omega`). Start there:

```bash
cd trm-deq-convergence-&-neumann-verification
cargo test --features cpu
```

Full build, measured numbers, and the implementation roadmap are in that README.
