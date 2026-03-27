#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/hostsfile.txt}"

VENV="${VENV:-$BASE_DIR/.venv}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"

NIC="${NIC:-enp193s0f0}"
TP="${TP:-2}"
PP="${PP:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
RAY_CPUS="${RAY_CPUS:-8}"

NUM_PROMPTS="${NUM_PROMPTS:-200}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
RATES="${RATES:-1 2 3 4}"

PORT="${PORT:-18084}"

RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-$(readlink -f /opt/rocm/core 2>/dev/null || true)}"
MPI_HOME="${MPI_HOME:-$BASE_DIR/openmpi}"

NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-VERSION}"
SYNC_RCCL="${SYNC_RCCL:-1}"

# ------------------------
# Parse hostfile
# ------------------------
mapfile -t HOSTS < <(awk '!/^[[:space:]]*($|#)/{print $1}' "$HOSTFILE")
[ "${#HOSTS[@]}" -ge 1 ] || { echo "Hostfile empty"; exit 1; }

# ------------------------
# Detect local host
# ------------------------
LOCAL_HOST_FULL="$(hostname)"
LOCAL_HOST_SHORT="$(hostname -s)"
mapfile -t LOCAL_IPS < <(hostname -I | tr ' ' '\n' | sed '/^$/d')

is_self() {
  local h="$1"
  [[ "$h" == "localhost" || "$h" == "127.0.0.1" ]] && return 0
  [[ "$h" == "$LOCAL_HOST_FULL" || "$h" == "$LOCAL_HOST_SHORT" ]] && return 0
  for ip in "${LOCAL_IPS[@]}"; do [[ "$h" == "$ip" ]] && return 0; done
  return 1
}

# ------------------------
# Build master + workers
# ------------------------
MASTER_HOST=""
WORKER_HOSTS=()

TOTAL_GPUS=$((GPUS_PER_NODE * (1 + ${#WORKER_HOSTS[@]})))
TP="${TP:-$TOTAL_GPUS}"

for host in "${HOSTS[@]}"; do
  if is_self "$host"; then
    MASTER_HOST="$host"
  else
    WORKER_HOSTS+=("$host")
  fi
done

if [ -z "$MASTER_HOST" ]; then
  MASTER_HOST="${HOSTS[0]}"
  WORKER_HOSTS=("${HOSTS[@]:1}")
  echo "[WARN] Local host not found, using $MASTER_HOST as master"
fi

echo "MASTER_HOST=$MASTER_HOST"
echo "WORKERS=${WORKER_HOSTS[*]}"

TOTAL_GPUS=$((GPUS_PER_NODE * (1 + ${#WORKER_HOSTS[@]})))
TP="${TP:-$TOTAL_GPUS}"

# ------------------------
# Cleanup trap
# ------------------------
trap 'cd "$SCRIPT_DIR"; BASE_DIR="$BASE_DIR" HOSTFILE="$HOSTFILE" ./stop_vllm_ssh.sh || true' EXIT

echo "Using VENV=$VENV"
echo "Using MODEL=$MODEL"

cd "$SCRIPT_DIR"

# ------------------------
# Setup
# ------------------------
BASE_DIR="$BASE_DIR" \
HOSTFILE="$HOSTFILE" \
VENV="$VENV" \
RCCL_HOME="$RCCL_HOME" \
ROCM_PATH="$ROCM_PATH" \
MPI_HOME="$MPI_HOME" \
SYNC_RCCL="$SYNC_RCCL" \
./setup_vllm_ssh.sh

# ------------------------
# Stop any previous run
# ------------------------
BASE_DIR="$BASE_DIR" \
HOSTFILE="$HOSTFILE" \
./stop_vllm_ssh.sh || true

# ------------------------
# Start cluster
# ------------------------
BASE_DIR="$BASE_DIR" \
HOSTFILE="$HOSTFILE" \
VENV="$VENV" \
MODEL="$MODEL" \
PORT="$PORT" \
TP="$TP" \
PP="$PP" \
GPUS_PER_NODE="$GPUS_PER_NODE" \
RAY_CPUS="$RAY_CPUS" \
IFACE="$NIC" \
WORKER_IFACE="$NIC" \
SOCKET_IFNAME="$NIC" \
WORKER_SOCKET_IFNAME="$NIC" \
RCCL_HOME="$RCCL_HOME" \
ROCM_PATH="$ROCM_PATH" \
MPI_HOME="$MPI_HOME" \
NCCL_DEBUG_LEVEL="$NCCL_DEBUG_LEVEL" \
./start_vllm_multinode_ssh.sh

# ------------------------
# Run benchmark
# ------------------------
BASE_DIR="$BASE_DIR" \
HOSTFILE="$HOSTFILE" \
VENV="$VENV" \
MODEL="$MODEL" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
PORT="$PORT" \
NUM_PROMPTS="$NUM_PROMPTS" \
OUTPUT_LEN="$OUTPUT_LEN" \
RANDOM_INPUT_LEN="$RANDOM_INPUT_LEN" \
RATES="$RATES" \
RCCL_HOME="$RCCL_HOME" \
ROCM_PATH="$ROCM_PATH" \
MPI_HOME="$MPI_HOME" \
NCCL_SOCKET_IFNAME="$NIC" \
NCCL_DEBUG_LEVEL="$NCCL_DEBUG_LEVEL" \
./run_sweep_ssh.sh

# ------------------------
# Final cleanup
# ------------------------
BASE_DIR="$BASE_DIR" \
HOSTFILE="$HOSTFILE" \
./stop_vllm_ssh.sh

echo "BENCHMARK_COMPLETED"