#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/hostsfile.txt}"  # List of all nodes: head + workers
NIC="${NIC:-enp193s0f0}"
VENV="${VENV:-$BASE_DIR/.venv}"
SKIP_BUILD_IF_PRESENT="${SKIP_BUILD_IF_PRESENT:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
RATES="${RATES:-1 2 3 8 16}"

SERVER_BIN="${SERVER_BIN:-$BASE_DIR/llama.cpp/build/bin/llama-server}"
RPC_SERVER_BIN="${RPC_SERVER_BIN:-$BASE_DIR/llama.cpp/build/bin/rpc-server}"

trap 'cd "$SCRIPT_DIR"; ./stop_llamacpp_ssh.sh "${HOSTS[@]}" || true' EXIT

# ------------------------
# Load hosts
# ------------------------

WORKERS=("${HOSTS[@]:1}")


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



echo "[INFO] Master node: $MASTER_NODE"
echo "[INFO] Worker nodes: ${WORKER_NODES[*]}"

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

# ------------------------
# Build llama.cpp on all nodes
# ------------------------
build_on_host() {
  local host="$1"
  if [ "$SKIP_BUILD_IF_PRESENT" = "1" ]; then
    if ssh $SSH_OPTS "$host" "bash -lc 'test -x \"$BASE_DIR/llama.cpp/build/bin/llama-server\" && test -x \"$BASE_DIR/llama.cpp/build/bin/rpc-server\"'"; then
      echo "== Skipping build on $host (binaries already present) =="
      return 0
    fi
  fi
  ssh $SSH_OPTS "$host" "bash -lc '
    set -euo pipefail
    BASE_DIR=\"$BASE_DIR\"
    cd \"\$BASE_DIR\"
    if [ ! -d llama.cpp ]; then git clone https://github.com/ggml-org/llama.cpp.git; fi
    cd llama.cpp
    cmake -S . -B build -DGGML_HIPBLAS=ON -DGGML_RPC=ON -DLLAMA_BUILD_BORINGSSL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target llama-server -j4
    cmake --build build --target rpc-server -j4 || cmake --build build --target llama-rpc-server -j4
  '"
}

echo "== Building llama.cpp on $MASTER_NODE =="
build_on_host "$MASTER_NODE"
for w in "${WORKER_NODES[@]}"; do
  echo "== Building llama.cpp on $w =="
  build_on_host "$w"
done

# ------------------------
# Ensure binaries exist
# ------------------------
ssh $SSH_OPTS "$MASTER_NODE" "bash -lc 'test -x \"$SERVER_BIN\"'" || { echo "[ERROR] llama-server missing on head"; exit 2; }

for w in "${WORKER_NODES[@]}"; do
  if ssh $SSH_OPTS "$w" "bash -lc 'test -x \"$RPC_SERVER_BIN\"'"; then
    :
  elif ssh $SSH_OPTS "$w" "bash -lc 'test -x \"$BASE_DIR/llama.cpp/build/bin/llama-rpc-server\"'"; then
    RPC_SERVER_BIN="$BASE_DIR/llama.cpp/build/bin/llama-rpc-server"
  else
    echo "[ERROR] RPC server missing on worker $w"; exit 2
  fi
done

# ------------------------
# Prepare benchmark scripts
# ------------------------
ssh $SSH_OPTS "$MASTER_NODE" "bash -lc 'mkdir -p \"$BASE_DIR/vllm-multinode\"'"
scp $SSH_OPTS "$SCRIPT_DIR/llamacpp_simple_bench.py" "$MASTER_NODE:$BASE_DIR/vllm-multinode/"

# ------------------------
# Start multiworker llama.cpp
# ------------------------
# Export common env for multiworker
export BASE_DIR VENV SERVER_BIN RPC_SERVER_BIN NIC
export USE_RPC=1 USE_MODULES=0 ACTIVATE_VENV=1 EXPORT_RCCL_ENV=1
export NET_IFACE="$NIC" WORKER_NET_IFACE="$NIC"
export NCCL_DEBUG_LEVEL=VERSION
export PARALLEL=1 CTX_SIZE=4096
export HF_REPO="TheBloke/Llama-2-7B-GGUF:Q4_K_M"
export MODEL=""
export MODEL_ALIAS="llama-2-7b-q4km"

# Start multi-node llama server
HOSTFILE="${HOSTFILE}" "$SCRIPT_DIR/start_llamacpp_multinode_ssh.sh"

# ------------------------
# Run multiworker sweep benchmark
# ------------------------
export TOKENIZER="NousResearch/Llama-2-7b-hf"
export BENCH_MODEL="NousResearch/Llama-2-7b-hf"
export MODEL_ALIAS="llama-2-7b-q4km"
export NUM_PROMPTS="$NUM_PROMPTS"
export OUTPUT_LEN="$OUTPUT_LEN"
export RATES="$RATES"
export BENCH_IMPL="simple"

HOSTFILE="${HOSTFILE}"  "$SCRIPT_DIR/run_llamacpp_sweep_ssh.sh"

# ------------------------
# Stop all llama.cpp servers
# ------------------------
HOSTFILE="${HOSTFILE}"  "$SCRIPT_DIR/stop_llamacpp_ssh.sh"

echo "BENCHMARK_COMPLETED"