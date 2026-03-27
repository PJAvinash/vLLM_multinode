#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$BASE_DIR/.venv}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
PORT="${PORT:-18084}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
RATES="${RATES:-1 2 3 4}"
OUTDIR="${OUTDIR:-$BASE_DIR/vllm_single_bench_results}"
LOGDIR="${LOGDIR:-$BASE_DIR/vllm_single_logs_$(date +%Y%m%d_%H%M%S)}"
RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-$(readlink -f /opt/rocm/core 2>/dev/null || true)}"
MPI_HOME="${MPI_HOME:-$BASE_DIR/openmpi}"
MPI_COMPAT_HOME="${MPI_COMPAT_HOME:-$BASE_DIR/mpi-compat}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp193s0f0}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-VERSION}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUTDIR" "$LOGDIR"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "Missing vLLM executable in $VENV/bin/vllm"
  echo "Run ./setup_vllm_ssh.sh first."
  exit 2
fi

cleanup() {
  set +u
  source /etc/profile >/dev/null 2>&1 || true
  set -u
  # shellcheck disable=SC1090
  source "$VENV/bin/activate" || true
  pkill -u "$USER" -f "[/]vllm.*serve" || true
  ray stop --force || true
}
trap cleanup EXIT

set +u
source /etc/profile >/dev/null 2>&1 || true
set -u
# shellcheck disable=SC1090
source "$VENV/bin/activate"
if [ -n "$ROCM_PATH" ] && [ -d "$ROCM_PATH/lib" ]; then
  export LD_LIBRARY_PATH="$MPI_COMPAT_HOME/lib:$MPI_HOME/lib:$RCCL_HOME:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$MPI_COMPAT_HOME/lib:$MPI_HOME/lib:$RCCL_HOME:${LD_LIBRARY_PATH:-}"
fi
export HIP_PLATFORM=amd
export PATH="$MPI_HOME/bin:$PATH"
export NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_DEBUG="$NCCL_DEBUG_LEVEL"
export HSA_NO_SCRATCH_RECLAIM=1

ray stop --force || true
pkill -u "$USER" -f "[/]vllm.*serve" || true

echo "Starting single-node vLLM serve on port $PORT ..."
nohup vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enforce-eager \
  > "$LOGDIR/vllm_serve.log" 2>&1 < /dev/null &
echo $! > "$LOGDIR/vllm_serve.pid"

echo "Waiting for /health ..."
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "vLLM did not become healthy. Recent logs:"
  tail -n 120 "$LOGDIR/vllm_serve.log" 2>/dev/null || true
  exit 3
fi

for RPS in $RATES; do
  echo "=== single-node RPS $RPS ==="
  vllm bench serve \
    --backend openai \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$RPS" \
    --disable-tqdm \
    --save-result \
    --result-dir "$OUTDIR" \
    --result-filename "single_rps${RPS}.json"

  "$PYTHON_BIN" - <<PY
import json
import sys
path = "$OUTDIR/single_rps${RPS}.json"
data = json.load(open(path))
completed = int(data.get("completed", 0))
failed = int(data.get("failed", 0))
print(f"completed={completed} failed={failed}")
sys.exit(0 if completed > 0 else 2)
PY
done

echo "Single-node benchmark completed."
echo "Results: $OUTDIR/single_rps*.json"
echo "Server logs: $LOGDIR/vllm_serve.log"
