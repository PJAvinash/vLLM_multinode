#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Config
# ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/../configs/hostsfile.txt}"

VENV="${VENV:-$BASE_DIR/.venv}"
LLAMA_CPP_BIN_DIR="${LLAMA_CPP_BIN_DIR:-$BASE_DIR/llama.cpp/build/bin}"
SERVER_BIN="${SERVER_BIN:-$LLAMA_CPP_BIN_DIR/llama-server}"
RPC_SERVER_BIN="${RPC_SERVER_BIN:-$LLAMA_CPP_BIN_DIR/rpc-server}"

MODEL="${MODEL:-}"                    # Either local GGUF path
HF_REPO="${HF_REPO:-}"                # Or HF repo
HF_FILE="${HF_FILE:-}"                # Optional HF file
MODEL_ALIAS="${MODEL_ALIAS:-llama-cpp-model}"

PORT="${PORT:-18085}"
RPC_PORT="${RPC_PORT:-50052}"
USE_RPC="${USE_RPC:-1}"               # 1=enable worker RPC, 0=skip
PARALLEL="${PARALLEL:-8}"
N_GPU_LAYERS="${N_GPU_LAYERS:-all}"
CTX_SIZE="${CTX_SIZE:-8192}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
ACTIVATE_VENV="${ACTIVATE_VENV:-1}"
USE_MODULES="${USE_MODULES:-0}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-ubuntu-24 rocm/7.1.1}"
EXPORT_RCCL_ENV="${EXPORT_RCCL_ENV:-1}"
RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-$(readlink -f /opt/rocm/core 2>/dev/null || true)}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-VERSION}"

LOGDIR="${LOGDIR:-$BASE_DIR/logs/llamacpp_logs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGDIR"

SSH_USER="${SSH_USER:-$USER}"

# ------------------------
# Host detection
# ------------------------
mapfile -t HOSTS < <(awk '!/^[[:space:]]*($|#)/{print $1}' "$HOSTFILE")
[ "${#HOSTS[@]}" -ge 1 ] || { echo "[ERROR] Hostfile empty"; exit 1; }

LOCAL_HOST_FULL="$(hostname 2>/dev/null || true)"
LOCAL_HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
LOCAL_HOST_FQDN="$(hostname -f 2>/dev/null || true)"
mapfile -t LOCAL_IPS < <(hostname -I 2>/dev/null | tr ' ' '\n' | sed '/^$/d')

is_local_host() {
  local host="$1"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$LOCAL_HOST_FULL" || "$host" == "$LOCAL_HOST_SHORT" || "$host" == "$LOCAL_HOST_FQDN" ]] && return 0
  for ip in "${LOCAL_IPS[@]}"; do [[ "$host" == "$ip" ]] && return 0; done
  return 1
}

MASTER=""
WORKERS=()
for h in "${HOSTS[@]}"; do
  if is_local_host "$h"; then
    MASTER="$h"
  else
    WORKERS+=("$h")
  fi
done
[ -n "$MASTER" ] || MASTER="${HOSTS[0]}"  # fallback
[ "${#WORKERS[@]}" -eq 0 ] && echo "[WARN] Only master detected, no workers will be used"

echo "[INFO] Master: $MASTER"
echo "[INFO] Workers: ${WORKERS[*]}"

MASTER_SSH="${SSH_USER}@${MASTER}"
WORKER_SSHS=()
for w in "${WORKERS[@]}"; do
  WORKER_SSHS+=("${SSH_USER}@${w}")
done

ssh_cmd() {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o ServerAliveInterval=10 -o ServerAliveCountMax=6 "$@"
}

# ------------------------
# Environment preparation
# ------------------------
PREP_MODULES="true"
if [ "$USE_MODULES" = "1" ]; then
  PREP_MODULES="source /etc/profile.d/modules.sh >/dev/null 2>&1 || true; \
                if command -v module >/dev/null 2>&1; then module purge || true; \
                for mod in $MODULES_TO_LOAD; do module load \"\$mod\"; done; fi"
fi

PREP_VENV="true"
if [ "$ACTIVATE_VENV" = "1" ]; then
  PREP_VENV="if [ -d \"$VENV\" ]; then source \"$VENV/bin/activate\"; fi"
fi

PREP_COMMON="source /etc/profile >/dev/null 2>&1 || true; $PREP_MODULES; $PREP_VENV; ulimit -n 65535 >/dev/null 2>&1 || true"
PREP_RCCL=":"
if [ "$EXPORT_RCCL_ENV" = "1" ]; then
  PREP_RCCL="export HIP_PLATFORM=amd; export NCCL_DEBUG=$NCCL_DEBUG_LEVEL; \
              if [ -n \"$ROCM_PATH\" ] && [ -d \"$ROCM_PATH/lib\" ]; then export LD_LIBRARY_PATH=\"$ROCM_PATH/lib:\${LD_LIBRARY_PATH:-}\"; fi; \
              if [ -n \"$RCCL_HOME\" ] && [ -d \"$RCCL_HOME\" ]; then export RCCL_HOME=\"$RCCL_HOME\"; export LD_LIBRARY_PATH=\"$RCCL_HOME:\${LD_LIBRARY_PATH:-}\"; fi"
fi

run_on_master() {
  local cmd="$1"
  if is_local_host "$MASTER"; then
    bash -lc "$cmd"
  else
    ssh_cmd "$MASTER_SSH" "bash -lc '$cmd'"
  fi
}

run_on_worker() {
  local worker="$1"
  local cmd="$2"
  ssh_cmd "$SSH_USER@$worker" "bash -lc '$cmd'"
}

# ------------------------
# Model args
# ------------------------
if [ -n "$MODEL" ]; then
  MODEL_ARGS="--model \"$MODEL\""
  MODEL_DESC="$MODEL"
elif [ -n "$HF_REPO" ]; then
  MODEL_ARGS="--hf-repo \"$HF_REPO\""
  MODEL_DESC="$HF_REPO"
  [ -n "$HF_FILE" ] && { MODEL_ARGS="$MODEL_ARGS --hf-file \"$HF_FILE\""; MODEL_DESC="$HF_REPO (file: $HF_FILE)"; }
else
  echo "[ERROR] Set either MODEL=/path/to/model.gguf or HF_REPO=org/repo:quant"
  exit 2
fi

# ------------------------
# Auto detect network interfaces
# ------------------------
if [ -z "${NET_IFACE:-}" ]; then
  NET_IFACE=$(run_on_master "ip -4 -o addr show scope global | awk '\$4 !~ /^169\\.254\\./ && \$2 !~ /^(lo|docker|virbr|cni|flannel|veth|usb)/ {print \$2; exit}'")
fi
[ -n "$NET_IFACE" ] || { echo "[ERROR] Could not auto-detect master interface"; exit 1; }
MASTER_IP=$(run_on_master "ip -4 -o addr show dev $NET_IFACE primary scope global | sed -n '1s/.* inet \\([0-9.]*\\)\\/.*$/\\1/p'")
[ -n "$MASTER_IP" ] || { echo "[ERROR] Could not detect MASTER_IP on $NET_IFACE"; exit 1; }

# Workers auto-detect interfaces if RPC enabled
WORKER_IPS=()
WORKER_IFACES=()
if [ "$USE_RPC" = "1" ]; then
  for w in "${WORKERS[@]}"; do
    iface=$(ssh_cmd "$SSH_USER@$w" "ip -4 -o addr show scope global | awk '\$4 !~ /^169\\.254\\./ && \$2 !~ /^(lo|docker|virbr|cni|flannel|veth|usb)/ {print \$2; exit}'")
    ip=$(ssh_cmd "$SSH_USER@$w" "ip -4 -o addr show dev $iface primary scope global | sed -n '1s/.* inet \\([0-9.]*\\)\\/.*$/\\1/p'")
    [ -n "$ip" ] || { echo "[ERROR] Could not detect WORKER_IP on $w"; exit 1; }
    WORKER_IPS+=("$ip")
    WORKER_IFACES+=("$iface")
  done
fi

# ------------------------
# Start worker RPC servers
# ------------------------
if [ "$USE_RPC" = "1" ]; then
  for idx in "${!WORKERS[@]}"; do
    w="${WORKERS[$idx]}"
    w_ip="${WORKER_IPS[$idx]}"
    w_iface="${WORKER_IFACES[$idx]}"
    echo "[INFO] Starting RPC server on worker $w ($w_ip)"
    run_on_worker "$w" "$PREP_COMMON; $PREP_RCCL; export NCCL_SOCKET_IFNAME=$w_iface; mkdir -p $LOGDIR; pkill -u $USER -x rpc-server || true; nohup $RPC_SERVER_BIN -H 0.0.0.0 -p $RPC_PORT > $LOGDIR/rpc_${w}.log 2>&1 < /dev/null & echo \$! > $LOGDIR/rpc_${w}.pid"
  done

  echo "[INFO] Waiting for all worker RPC endpoints..."
  for w_ip in "${WORKER_IPS[@]}"; do
    for _ in $(seq 1 120); do
      if run_on_master "echo > /dev/tcp/$w_ip/$RPC_PORT" >/dev/null 2>&1; then break; fi
      sleep 2
    done
  done
fi

# ------------------------
# Start master llama-server
# ------------------------
# Construct a comma-separated list of worker_ip:port
if [ "$USE_RPC" = "1" ] && [ ${#WORKER_IPS[@]} -gt 0 ]; then
    RPC_ENDPOINTS=()
    for ip in "${WORKER_IPS[@]}"; do
        RPC_ENDPOINTS+=("$ip:$RPC_PORT")
    done
    # Join with commas
    RPC_ARG="--rpc $(IFS=,; echo "${RPC_ENDPOINTS[*]}")"
else
    RPC_ARG=""
fi

echo "[INFO] Starting llama-server on master $MASTER"
run_on_master "$PREP_COMMON; $PREP_RCCL; export NCCL_SOCKET_IFNAME=$NET_IFACE; mkdir -p $LOGDIR; pkill -u $USER -x llama-server || true; nohup $SERVER_BIN $MODEL_ARGS --alias $MODEL_ALIAS --host 0.0.0.0 --port $PORT $RPC_ARG --n-gpu-layers $N_GPU_LAYERS --ctx-size $CTX_SIZE --parallel $PARALLEL --metrics $SERVER_EXTRA_ARGS > $LOGDIR/llama_server.log 2>&1 < /dev/null & echo \$! > $LOGDIR/llama_server.pid"

# ------------------------
# Health check
# ------------------------
echo "[INFO] Waiting for llama-server health and model routes..."
for _ in $(seq 1 300); do
  if run_on_master "curl -sf http://127.0.0.1:$PORT/health >/dev/null" && \
     run_on_master "curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null"; then
    echo "READY: http://$MASTER:$PORT/health"
    echo "OPENAI: http://$MASTER:$PORT/v1/completions"
    echo "LOGDIR: $LOGDIR"
    exit 0
  fi
  sleep 3
done

echo "[ERROR] llama-server did not become healthy. Recent logs:"
run_on_master "tail -n 120 $LOGDIR/llama_server.log 2>/dev/null || true"

if [ "$USE_RPC" = "1" ]; then
  for idx in "${!WORKERS[@]}"; do
    w="${WORKERS[$idx]}"
    echo "Worker RPC log tail ($w):"
    run_on_worker "$w" "tail -n 120 $LOGDIR/rpc_${w}.log 2>/dev/null || true"
  done
fi

exit 4