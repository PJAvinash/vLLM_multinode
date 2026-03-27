#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Config
# ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"

HOSTFILE="${HOSTFILE:-$BASE_DIR/../configs/hostsfile.txt}"
SSH_USER="${SSH_USER:-$USER}"

VENV="${VENV:-$BASE_DIR/.venv}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

IFACE="${IFACE:-auto}"
WORKER_IFACE="${WORKER_IFACE:-$IFACE}"

PORT="${PORT:-18084}"
TP="${TP:-2}"
PP="${PP:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

RAY_CPUS="${RAY_CPUS:-8}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
RAY_CGRAPH_GET_TIMEOUT="${RAY_CGRAPH_GET_TIMEOUT:-1800}"

RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
MPI_HOME="${MPI_HOME:-${BASE_DIR}/mpich/install}"

NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-VERSION}"

LOGDIR="${LOGDIR:-$BASE_DIR/logs/vllm_logs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGDIR"

# ------------------------
# Parse hostfile
# ------------------------
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
MASTER_HOST=""
WORKER_HOSTS=()

for host in "${HOSTS[@]}"; do
    if is_self "$host"; then
        MASTER_HOST="$host"
    else
        WORKER_HOSTS+=("$host")
    fi
done

# fallback if local not found
if [ -z "$MASTER_HOST" ]; then
    MASTER_HOST="${HOSTS[0]}"
    WORKER_HOSTS=("${HOSTS[@]:1}")
    echo "[WARN] Local host not in hostfile, using $MASTER_HOST as master"
fi

echo "[INFO] Master: $MASTER_HOST"
echo "[INFO] Workers: ${WORKER_HOSTS[*]}"

TOTAL_GPUS=$((GPUS_PER_NODE * (1 + ${#WORKER_HOSTS[@]})))
TP="${TP:-$TOTAL_GPUS}"
# ------------------------
# SSH helper
# ------------------------
ssh_cmd() {
  ssh -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=20 \
      "$@"
}

run_on_head() {
  if is_self "$MASTER_HOST"; then
    bash -lc "$1"
  else
    ssh_cmd "${SSH_USER}@${MASTER_HOST}" "bash -lc $(printf '%q' "$1")"
  fi
}

# ------------------------
# Detect HEAD iface + IP
# ------------------------
if [ "$IFACE" = "auto" ]; then
  IFACE="$(run_on_head "ip -4 -o addr show scope global | awk '\$4 !~ /^169\\.254/ && \$2 !~ /^(lo|docker|virbr|cni|veth)/ {print \$2; exit}'")"
fi

HEAD_IP="$(run_on_head "ip -4 -o addr show dev $IFACE primary | sed -n '1s/.* inet \\([0-9.]*\\)\\/.*$/\\1/p'")"

[ -n "$HEAD_IP" ] || { echo "Failed to detect HEAD IP"; exit 1; }

# ------------------------
# Detect worker iface + IPs
# ------------------------
declare -A WORKER_IP_MAP
declare -A WORKER_IFACE_MAP

for w in "${WORKER_HOSTS[@]}"; do
  target="${SSH_USER}@${w}"

  if [ "$WORKER_IFACE" = "auto" ]; then
    WORKER_IFACE_MAP["$w"]=$(ssh_cmd "$target" \
      "ip -4 -o addr show scope global | awk '\$4 !~ /^169\\.254/ && \$2 !~ /^(lo|docker|virbr|cni|veth)/ {print \$2; exit}'")
  else
    WORKER_IFACE_MAP["$w"]="$WORKER_IFACE"
  fi

  WORKER_IP_MAP["$w"]=$(ssh_cmd "$target" \
    "ip -4 -o addr show dev ${WORKER_IFACE_MAP[$w]} primary | sed -n '1s/.* inet \\([0-9.]*\\)\\/.*$/\\1/p'")

  [ -n "${WORKER_IP_MAP[$w]}" ] || { echo "Failed to detect IP for $w"; exit 1; }
done

# ------------------------
# Common env
# ------------------------
PREP_COMMON="source /etc/profile >/dev/null 2>&1 || true; \
[ -d \"$VENV\" ] && source \"$VENV/bin/activate\"; \
ulimit -n 65535 || true; \
export HIP_PLATFORM=amd; \
export PATH=\"$MPI_HOME/bin:\$PATH\"; \
export NCCL_DEBUG=$NCCL_DEBUG_LEVEL; \
export RCCL_HOME=\"$RCCL_HOME\"; \
export ROCM_PATH=\"$ROCM_PATH\"; \
export LD_LIBRARY_PATH=\"$MPI_HOME/lib:$RCCL_HOME:$ROCM_PATH/lib:\${LD_LIBRARY_PATH:-}\""

# ------------------------
# Start Ray head
# ------------------------
echo "[1/4] Starting Ray head ($MASTER_HOST)"
run_on_head "$PREP_COMMON; \
export NCCL_SOCKET_IFNAME=$IFACE; \
export VLLM_HOST_IP=$HEAD_IP; \
ray stop --force || true; \
ray start --head --node-ip-address $HEAD_IP --port 6379 \
--num-gpus $GPUS_PER_NODE --num-cpus $RAY_CPUS \
--include-dashboard=$RAY_INCLUDE_DASHBOARD"

# ------------------------
# Start Ray workers
# ------------------------
for w in "${WORKER_HOSTS[@]}"; do
  ip="${WORKER_IP_MAP[$w]}"
  iface="${WORKER_IFACE_MAP[$w]}"

  echo "[2/4] Worker $w ($ip)"

  ssh_cmd "${SSH_USER}@${w}" "bash -lc '$PREP_COMMON; \
    export NCCL_SOCKET_IFNAME=$iface; \
    export VLLM_HOST_IP=$ip; \
    ray stop --force || true; \
    ray start --address $HEAD_IP:6379 \
               --node-ip-address $ip \
               --num-gpus $GPUS_PER_NODE \
               --num-cpus $RAY_CPUS'"
done

# ------------------------
# Start vLLM
# ------------------------
echo "[3/4] Starting vLLM on head"
run_on_head "$PREP_COMMON; \
export NCCL_SOCKET_IFNAME=$IFACE; \
export VLLM_HOST_IP=$HEAD_IP; \
mkdir -p \"$LOGDIR\"; \
nohup vllm serve \"$MODEL\" \
  --host 0.0.0.0 --port $PORT \
  --distributed-executor-backend ray \
  --tensor-parallel-size $TP \
  --pipeline-parallel-size $PP \
  > \"$LOGDIR/vllm.log\" 2>&1 &"

# ------------------------
# Health check
# ------------------------
echo "[4/4] Waiting for service..."
for _ in $(seq 1 120); do
  if run_on_head "curl -sf http://127.0.0.1:$PORT/health >/dev/null"; then
    echo "READY: http://$MASTER_HOST:$PORT/health"
    exit 0
  fi
  sleep 2
done

echo "FAILED: Check logs at $LOGDIR"
exit 1