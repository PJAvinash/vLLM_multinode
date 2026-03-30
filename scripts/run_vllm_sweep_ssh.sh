#!/usr/bin/env bash
set -euo pipefail

HEAD_HOST="${1:?usage: $0 <head-host>}"
SSH_USER="${SSH_USER:-$USER}"
HEAD_TARGET_HOST="${HEAD_SSH_HOST:-$HEAD_HOST}"
HEAD="${SSH_USER}@${HEAD_TARGET_HOST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$BASE_DIR/.venv}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
DATASET_NAME="${DATASET_NAME:-random}"
DATASET="${DATASET:-$BASE_DIR/ShareGPT_V3_unfiltered_cleaned_split.json}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
PORT="${PORT:-18084}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RATES="${RATES:-1 2 3 8 16}"
OUTDIR="${OUTDIR:-$BASE_DIR/bench_results}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-900}"
HEALTH_POLL_SEC="${HEALTH_POLL_SEC:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-$(readlink -f /opt/rocm/core 2>/dev/null || true)}"
MPI_HOME="${MPI_HOME:-$BASE_DIR/openmpi}"
MPI_COMPAT_HOME="${MPI_COMPAT_HOME:-$BASE_DIR/mpi-compat}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp193s0f0}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-VERSION}"

ssh_cmd() {
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=20 \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=6 \
    "$@"
}

LOCAL_HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
LOCAL_HOST_FULL="$(hostname 2>/dev/null || true)"
LOCAL_HOST_FQDN="$(hostname -f 2>/dev/null || true)"
mapfile -t LOCAL_IPS < <(hostname -I 2>/dev/null | tr ' ' '\n' | sed '/^$/d')

is_local_target() {
  local target="$1"
  local target_short="${target%%.*}"
  if [ "$target" = "localhost" ] || [ "$target" = "127.0.0.1" ] || \
     [ "$target" = "$LOCAL_HOST_SHORT" ] || [ "$target" = "$LOCAL_HOST_FULL" ] || \
     [ "$target" = "$LOCAL_HOST_FQDN" ] || [ "$target_short" = "$LOCAL_HOST_SHORT" ]; then
    return 0
  fi
  local ip
  for ip in "${LOCAL_IPS[@]}"; do
    [ "$target" = "$ip" ] && return 0
  done
  return 1
}

run_on_head() {
  if is_local_target "$HEAD_HOST" || is_local_target "$HEAD_TARGET_HOST"; then
    bash -lc "$1"
  else
    local escaped
    escaped=$(printf '%q' "$1")
    ssh_cmd "$HEAD" "bash -lc $escaped"
  fi
}

echo "Waiting for vLLM health on 127.0.0.1:$PORT ..."
MAX_TRIES=$((HEALTH_TIMEOUT_SEC / HEALTH_POLL_SEC))
for _ in $(seq 1 "$MAX_TRIES"); do
  if run_on_head "curl -sf http://127.0.0.1:$PORT/health >/dev/null"; then
    echo "vLLM is healthy."
    break
  fi
  sleep "$HEALTH_POLL_SEC"
done

if ! run_on_head "curl -sf http://127.0.0.1:$PORT/health >/dev/null"; then
  echo "vLLM did not become healthy within ${HEALTH_TIMEOUT_SEC}s."
  LATEST_LOGDIR="$(run_on_head "ls -td $BASE_DIR/vllm_logs_* 2>/dev/null | head -n1" || true)"
  if [ -n "${LATEST_LOGDIR:-}" ]; then
    echo "Latest log dir: $LATEST_LOGDIR"
    run_on_head "tail -n 80 $LATEST_LOGDIR/vllm_serve.log 2>/dev/null || true"
  fi
  exit 7
fi

echo "Waiting for vLLM inference readiness on 127.0.0.1:$PORT ..."
if ! run_on_head "
  source /etc/profile >/dev/null 2>&1 || true
  source $VENV/bin/activate
  \"$PYTHON_BIN\" - <<'PY'
import json
import sys
import time
import urllib.request

port = int(\"$PORT\")
model = \"$SERVED_MODEL_NAME\"
deadline = time.time() + int(\"$HEALTH_TIMEOUT_SEC\")
poll = max(1, int(\"$HEALTH_POLL_SEC\"))
url = f\"http://127.0.0.1:{port}/v1/completions\"
payload = json.dumps({
    \"model\": model,
    \"prompt\": \"ping\",
    \"max_tokens\": 1,
    \"temperature\": 0,
}).encode(\"utf-8\")
headers = {\"Content-Type\": \"application/json\"}
last_error = \"no attempts\"

while time.time() < deadline:
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method=\"POST\")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode(\"utf-8\", errors=\"ignore\")
            if resp.status == 200 and '\"choices\"' in body:
                print(\"Inference endpoint is ready.\")
                sys.exit(0)
            last_error = f\"HTTP {resp.status}: {body[:200]}\"
    except Exception as exc:  # noqa: BLE001
        last_error = str(exc)
    time.sleep(poll)

print(f\"Timed out waiting for inference readiness: {last_error}\", file=sys.stderr)
sys.exit(1)
PY
"; then
  echo "vLLM did not become inference-ready within ${HEALTH_TIMEOUT_SEC}s."
  LATEST_LOGDIR="$(run_on_head "ls -td $BASE_DIR/vllm_logs_* 2>/dev/null | head -n1" || true)"
  if [ -n "${LATEST_LOGDIR:-}" ]; then
    echo "Latest log dir: $LATEST_LOGDIR"
    run_on_head "tail -n 120 $LATEST_LOGDIR/vllm_serve.log 2>/dev/null || true"
  fi
  exit 10
fi

if [ "$DATASET_NAME" = "sharegpt" ] && ! run_on_head "test -f \"$DATASET\""; then
  echo "DATASET_NAME=sharegpt but dataset file not found: $DATASET"
  exit 6
fi

for RPS in $RATES; do
  echo "=== RPS $RPS ==="
  if ! run_on_head "curl -sf http://127.0.0.1:$PORT/health >/dev/null"; then
    echo "Server is not healthy before running RPS $RPS; aborting."
    LATEST_LOGDIR="$(run_on_head "ls -td $BASE_DIR/vllm_logs_* 2>/dev/null | head -n1" || true)"
    if [ -n "${LATEST_LOGDIR:-}" ]; then
      echo "Latest log dir: $LATEST_LOGDIR"
      run_on_head "tail -n 120 $LATEST_LOGDIR/vllm_serve.log 2>/dev/null || true"
    fi
    exit 8
  fi

  run_on_head "
    source /etc/profile >/dev/null 2>&1 || true
    source $VENV/bin/activate
    export HIP_PLATFORM=amd
    export PATH=\"$MPI_HOME/bin:\$PATH\"
    export MPI_COMPAT_HOME=$MPI_COMPAT_HOME
    export RCCL_HOME=$RCCL_HOME
    export ROCM_PATH=$ROCM_PATH
    export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
    export NCCL_DEBUG=$NCCL_DEBUG_LEVEL
    if [ -n \"$ROCM_PATH\" ] && [ -d \"$ROCM_PATH/lib\" ]; then
      export LD_LIBRARY_PATH=\"$MPI_COMPAT_HOME/lib:$MPI_HOME/lib:$RCCL_HOME:$ROCM_PATH/lib:\${LD_LIBRARY_PATH:-}\"
    else
      export LD_LIBRARY_PATH=\"$MPI_COMPAT_HOME/lib:$MPI_HOME/lib:$RCCL_HOME:\${LD_LIBRARY_PATH:-}\"
    fi
    mkdir -p $OUTDIR
    DATASET_ARGS=\"\"
    if [ \"$DATASET_NAME\" = \"sharegpt\" ]; then
      DATASET_ARGS=\"--dataset-name sharegpt --dataset-path $DATASET\"
    else
      DATASET_ARGS=\"--dataset-name random --random-input-len $RANDOM_INPUT_LEN --random-output-len $OUTPUT_LEN\"
    fi
    vllm bench serve \
      --backend openai \
      --host 127.0.0.1 \
      --port $PORT \
      --model $MODEL \
      --served-model-name $SERVED_MODEL_NAME \
      --endpoint /v1/completions \
      --num-prompts $NUM_PROMPTS \
      --request-rate $RPS \
      --disable-tqdm \
      --save-result \
      --result-dir $OUTDIR \
      \$DATASET_ARGS \
      --result-filename mn2_ray_rps${RPS}.json
  "

  if ! run_on_head "PYBIN=\"$PYTHON_BIN\"; command -v \"\$PYBIN\" >/dev/null 2>&1 || PYBIN=python; command -v \"\$PYBIN\" >/dev/null 2>&1 || { echo 'No python interpreter found for JSON check'; exit 3; }; \"\$PYBIN\" -c 'import json,sys; p=\"$OUTDIR/mn2_ray_rps${RPS}.json\"; d=json.load(open(p)); c=int(d.get(\"completed\",0)); f=int(d.get(\"failed\",0)); print(\"completed=%d failed=%d\" % (c, f)); sys.exit(0 if c>0 else 2)'"; then
    echo "RPS $RPS produced zero successful requests; stopping sweep."
    LATEST_LOGDIR="$(run_on_head "ls -td $BASE_DIR/vllm_logs_* 2>/dev/null | head -n1" || true)"
    if [ -n "${LATEST_LOGDIR:-}" ]; then
      echo "Latest log dir: $LATEST_LOGDIR"
      run_on_head "tail -n 120 $LATEST_LOGDIR/vllm_serve.log 2>/dev/null || true"
    fi
    exit 9
  fi
done
