#!/usr/bin/env bash
# Headless train path for a rented Linux GPU (Colab / RunPod / Vast / Lambda).
#
# Colab (GPU runtime):
#   !curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   import os; os.environ["PATH"] += ":/root/.cargo/bin"
#   %cd /content/TRM/trm_omega   # or clone first
#   !bash scripts/cloud_train.sh
#
# Stages (STAGE=...):
#   print  — instantiate 7m on CPU, print live param count, exit
#   smoke  — dim 32, sudoku seq 81, batch 1, 1 step  (proves Device::new_cuda)
#   paper  — dim 256 / 2.67M, same soak
#   7m     — dim 448 / ~7M, same soak
#   train  — 7m, no max-steps (real run; still seq 81 / batch 1)
#   forge  — TRMQ10 from CHECKPOINT_DIR/final.safetensors
#
# Do not copy this binary to a 1660. Forge the .safetensors, copy the file,
# build `server` with --features cuda on the 1660.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

STAGE="${STAGE:-print}"
DATA_DIR="${DATA_DIR:-$ROOT/artifacts/cloud_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/artifacts/cloud_ckpt}"
FEATURES="${FEATURES:-candle-cuda}"
export RUST_LOG="${RUST_LOG:-info}"

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR"

common_flags=(
  --task sudoku
  --data-dir "$DATA_DIR"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --batch-size 1
  --micro-batch 1
  --fp16=false
  --no-augment
  --epochs 1
)

echo "stage=$STAGE  features=$FEATURES  cuda_home=${CUDA_HOME:-}  cuda_path=${CUDA_PATH:-}"
command -v rustc >/dev/null || { echo "rustc missing; install rustup first"; exit 1; }
rustc --version

case "$STAGE" in
  print)
    cargo run --release --features cpu --bin train -- --preset 7m --task sudoku --print-params
    ;;
  smoke)
    cargo run --release --features "$FEATURES" --bin train -- \
      --preset tiny \
      "${common_flags[@]}" \
      --max-steps 1
    ;;
  paper)
    cargo run --release --features "$FEATURES" --bin train -- \
      --preset paper \
      "${common_flags[@]}" \
      --max-steps 1
    ;;
  7m)
    cargo run --release --features "$FEATURES" --bin train -- \
      --preset 7m \
      "${common_flags[@]}" \
      --max-steps 1
    ;;
  train)
    cargo run --release --features "$FEATURES" --bin train -- \
      --preset 7m \
      "${common_flags[@]}" \
      --epochs "${EPOCHS:-50000}" \
      --eval-every "${EVAL_EVERY:-5000}"
    ;;
  forge)
    IN="${1:-$CHECKPOINT_DIR/final.safetensors}"
    OUT="${2:-$CHECKPOINT_DIR/model.trmq10}"
    cargo run --release --bin forge -- \
      --input "$IN" \
      --output "$OUT" \
      --preset 7m \
      --max-seq 243
    echo "copy $OUT to the 1660, then:"
    echo "  cargo run --release --features cuda --bin server -- --model $OUT --kernels --preset 7m --seq-len 81 --batch 1 --iters 4"
    ;;
  *)
    echo "unknown STAGE=$STAGE (print|smoke|paper|7m|train|forge)"
    exit 2
    ;;
esac
