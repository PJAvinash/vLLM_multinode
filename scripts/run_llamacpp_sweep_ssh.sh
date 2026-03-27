#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Config & Cluster Parsing
# ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/../configs/hostsfile.txt}"

mapfile -t HOSTS < <(awk '!/^[[:space:]]*($|#)/{print $1}' "$HOSTFILE")
[ "${#HOSTS[@]}" -gt 0 ] || { echo "[ERROR] Hostfile empty"; exit 1; }

# ------------------------
# Detect local host
# ------------------------
LOCAL_HOST_FULL="$(hostname)"
LOCAL_HOST_SHORT="$(hostname -s)"
LOCAL_HOST_FQDN="$(hostname -f 2>/dev/null || true)"
mapfile -t LOCAL_IPS < <(hostname -I | tr ' ' '\n' | sed '/^$/d')

is_self() {
    local h="$1"
    [[ "$h" == "localhost" || "$h" == "127.0.0.1" ]] && return 0
    [[ "$h" == "$LOCAL_HOST_FULL" || "$h" == "$LOCAL_HOST_SHORT" || "$h" == "$LOCAL_HOST_FQDN" ]] && return 0
    for ip in "${LOCAL_IPS[@]}"; do [[ "$h" == "$ip" ]] && return 0; done
    return 1
}

# ------------------------
# Build master + workers
# ------------------------
MASTER_NODE=""
WORKER_NODES=()

for host in "${HOSTS[@]}"; do
    if is_self "$host"; then
        MASTER_NODE="$host"
    else
        WORKER_NODES+=("$host")
    fi
done

# ------------------------
# Required Environment & Benchmarking Args
# ------------------------
SSH_USER="${SSH_USER:-$USER}"
VENV="${VENV:-$BASE_DIR/.venv}"
DATASET="${DATASET:-$BASE_DIR/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="${PORT:-18085}"
MODEL_ALIAS="${MODEL_ALIAS:-llama-cpp-model}"
TOKENIZER="${TOKENIZER:-}"
BENCH_MODEL="${BENCH_MODEL:-$MODEL_ALIAS}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RATES="${RATES:-1 2 4 8 16}"
OUTDIR="${OUTDIR:-$BASE_DIR/llamacpp_bench_results}"
RESULT_PREFIX="${RESULT_PREFIX:-llamacpp}"

# Logic Control
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-900}"
HEALTH_POLL_SEC="${HEALTH_POLL_SEC:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_MODULES="${USE_MODULES:-0}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-ubuntu-24 rocm/7.1.1}"
ACTIVATE_VENV="${ACTIVATE_VENV:-1}"
BENCH_IMPL="${BENCH_IMPL:-auto}"

# ------------------------
# SSH & Environment Helpers
# ------------------------
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
ssh_cmd() { ssh $SSH_OPTS "$@"; }

run_on_host() {
    local host="$1"; shift
    local cmd="$*"
    ssh_cmd "${SSH_USER}@${host}" "bash -lc '$cmd'"
}

# Environment Prep String
PREP_MODULES=":"
if [ "$USE_MODULES" = "1" ]; then
    PREP_MODULES="source /etc/profile.d/modules.sh >/dev/null 2>&1 || true; if command -v module >/dev/null 2>&1; then module purge || true; for mod in $MODULES_TO_LOAD; do module load \"\$mod\"; done; fi;"
fi

PREP_VENV=":"
if [ "$ACTIVATE_VENV" = "1" ]; then
    PREP_VENV="if [ -d \"$VENV\" ]; then source \"$VENV/bin/activate\"; fi;"
fi

# ------------------------
# 1. Automatic Backend Detection
# ------------------------
if [ "$BENCH_IMPL" = "auto" ]; then
    if run_on_host "$MASTER_NODE" "$PREP_MODULES $PREP_VENV command -v vllm >/dev/null 2>&1"; then
        BENCH_IMPL="vllm"
    else
        BENCH_IMPL="simple"
    fi
fi
echo "[INFO] Using BENCH_IMPL=$BENCH_IMPL, Target: $MASTER_NODE"

# ------------------------
# 2. Distributed Health Check
# ------------------------
echo "[INFO] Checking cluster health on $MASTER_NODE..."
MAX_TRIES=$((HEALTH_TIMEOUT_SEC / HEALTH_POLL_SEC))
HEALTHY=0

for i in $(seq 1 "$MAX_TRIES"); do
    if run_on_host "$MASTER_NODE" "curl -sf http://127.0.0.1:$PORT/health >/dev/null"; then
        echo "[OK] Master node is healthy and RPC workers are connected."
        HEALTHY=1
        break
    fi
    sleep "$HEALTH_POLL_SEC"
done

if [ "$HEALTHY" -eq 0 ]; then
    echo "[ERROR] Cluster failed to stabilize within $HEALTH_TIMEOUT_SEC seconds."
    exit 7
fi

# ------------------------
# 3. Benchmark Sweep
# ------------------------
for RPS in $RATES; do
    echo "=== RPS $RPS ==="
    run_on_host "$MASTER_NODE" "
        $PREP_MODULES
        $PREP_VENV
        mkdir -p $OUTDIR
        if [ \"$BENCH_IMPL\" = \"vllm\" ]; then
            vllm bench serve \
                --backend openai --host 127.0.0.1 --port $PORT \
                --model $BENCH_MODEL --served-model-name $MODEL_ALIAS \
                ${TOKENIZER:+--tokenizer $TOKENIZER} \
                --dataset-path $DATASET --num-prompts $NUM_PROMPTS \
                --output-len $OUTPUT_LEN --request-rate $RPS \
                --save-result --result-dir $OUTDIR \
                --result-filename ${RESULT_PREFIX}_rps${RPS}.json
        else
            $PYTHON_BIN $BASE_DIR/vllm-multinode/llamacpp_simple_bench.py \
                --url http://127.0.0.1:$PORT/v1/completions \
                --model $MODEL_ALIAS --num-prompts $NUM_PROMPTS \
                --output-len $OUTPUT_LEN --request-rate $RPS \
                --result-file $OUTDIR/${RESULT_PREFIX}_rps${RPS}.json
        fi
    "
done

echo "BENCHMARK_COMPLETED"

# ------------------------
# 4. Generate Summary Table
# ------------------------
echo -e "\n=== BENCHMARK SUMMARY ($RESULT_PREFIX) ==="
printf "%-10s | %-12s | %-12s | %-12s\n" "RPS" "Completed" "Avg Latency" "Tokens/s"
echo "------------------------------------------------------------"

for RPS in $RATES; do
    FILE="$OUTDIR/${RESULT_PREFIX}_rps${RPS}.json"
    if [ -f "$FILE" ]; then
        python3 -c "
import json
try:
    with open('$FILE') as f:
        d = json.load(f)
    comp = d.get('completed', 0)
    lat = round(d.get('avg_latency_ms', 0), 2)
    tps = round(d.get('avg_tokens_per_sec', 0), 2)
    print(f'${RPS:<10} | {comp:<12} | {lat:<12} | {tps:<12}')
except Exception:
    print(f'${RPS:<10} | ERROR READING FILE')
"
    else
        printf "%-10s | %-12s | %-12s | %-12s\n" "$RPS" "MISSING" "-" "-"
    fi
done
echo "------------------------------------------------------------"

echo "[INFO] All detailed JSONs are located in: $OUTDIR"