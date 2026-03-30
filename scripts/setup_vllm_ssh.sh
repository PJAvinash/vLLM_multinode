#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/../configs/hostsfile.txt}"
VENV="${VENV:-$BASE_DIR/.venv}"
RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

MPI_HOME="${MPI_HOME:-${BASE_DIR}/mpich/install}"
MPI_COMPAT_HOME="${MPI_COMPAT_HOME:-/usr}"
VLLM_VERSION="${VLLM_VERSION:-0.18.0}"
VLLM_INDEX_URL="${VLLM_INDEX_URL:-https://wheels.vllm.ai/rocm}"
SYNC_RCCL="${SYNC_RCCL:-1}"
SSH_USER="${SSH_USER:-$USER}"
DEBUG_INSTALL="${DEBUG_INSTALL:-0}"

ssh_cmd() {
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=20 \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=6 \
    "$@"
}

mapfile -t HOSTS_FROM_FILE < <(awk '!/^[[:space:]]*($|#)/{print $1}' "$HOSTFILE")
if [ "$#" -gt 0 ]; then
  HOSTS=("$@")
else
  HOSTS=("${HOSTS_FROM_FILE[@]}")
fi

if [ "${#HOSTS[@]}" -eq 0 ]; then
  echo "No hosts provided and hostfile is empty: $HOSTFILE"
  exit 1
fi

LOCAL_HOST_FULL="$(hostname 2>/dev/null || true)"
LOCAL_HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
mapfile -t LOCAL_IPS < <(hostname -I 2>/dev/null | tr ' ' '\n' | sed '/^$/d')

is_local_host() {
  local host="$1"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$LOCAL_HOST_FULL" || "$host" == "$LOCAL_HOST_SHORT" ]] && return 0
  local ip
  for ip in "${LOCAL_IPS[@]}"; do
    [[ "$host" == "$ip" ]] && return 0
  done
  return 1
}

ensure_rccl_on_host() {
  local host="$1"
  if is_local_host "$host"; then
    echo "$RCCL_HOME"
    test -d "$RCCL_HOME"
    return 0
  fi
  
  if ssh_cmd "${SSH_USER}@${host}" "test -d \"$RCCL_HOME\""; then
    return 0
  fi

  if [ "$SYNC_RCCL" != "1" ]; then
    echo "RCCL directory missing on $host and SYNC_RCCL=0: $RCCL_HOME"
    return 1
  fi
}

ensure_mpi_compat_on_host() {
  local host="$1"
  local compat_lib="$MPI_COMPAT_HOME/lib"

  if [ ! -d "$compat_lib" ]; then
    echo "Missing local MPI compatibility directory: $compat_lib"
    return 1
  fi

  if is_local_host "$host"; then
    test -d "$compat_lib"
    return 0
  fi
}

setup_host() {
  local host="$1"
  local target="${SSH_USER}@${host}"
  
  # We use a heredoc but carefully escape variables we want evaluated REMOTELY
  local install_cmd=$(cat <<EOF
    set -euo pipefail
    mkdir -p "$BASE_DIR"
    cd "$BASE_DIR"

    # --- 1. IDEMPOTENCY CHECK ---
    # Don't reinstall if vLLM is already working on this node
    if [ -f "$VENV/bin/python" ]; then
        if "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
            echo "[INFO] vLLM already installed on $host. Skipping..."
            exit 0
        fi
    fi

    # --- 2. PYTHON SETUP ---
    # Simplified: Always try to create a venv at the specified \$VENV path
    PYTHON_EXE=""
    for cmd in python3.12 python3.11 python3.10 python3; do
        if command -v \$cmd >/dev/null 2>&1; then
            PYTHON_EXE=\$(command -v \$cmd)
            break
        fi
    done

    if [ -z "\$PYTHON_EXE" ]; then
        echo "[ERROR] No compatible Python found on $host"
        exit 1
    fi

    "\$PYTHON_EXE" -m venv "$VENV" --system-site-packages || "\$PYTHON_EXE" -m venv "$VENV"
    source "$VENV/bin/activate"

    # --- 3. INSTALLATION ---
    python -m pip install --upgrade pip setuptools wheel --no-cache-dir
    
    # Use --no-cache-dir to prevent filling up disk on worker nodes
    python -m pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2

    pip install /opt/rocm/share/amd_smi
    
    python -m pip install --no-cache-dir \
      "vllm==$VLLM_VERSION" --extra-index-url "$VLLM_INDEX_URL"

    # --- 4. PERSISTENT ENVIRONMENT (CRITICAL) ---
    # We append the library paths to the venv activation script itself
    ACTIVATE_LIB_STR="export LD_LIBRARY_PATH=\"${MPI_COMPAT_HOME}/lib:$MPI_HOME/lib:$RCCL_HOME:$ROCM_PATH/lib:\\\$LD_LIBRARY_PATH\""
    if ! grep -q "LD_LIBRARY_PATH" "$VENV/bin/activate"; then
        echo "export RCCL_HOME=\"$RCCL_HOME\"" >> "$VENV/bin/activate"
        echo "export ROCM_PATH=\"$ROCM_PATH\"" >> "$VENV/bin/activate"
        echo "\$ACTIVATE_LIB_STR" >> "$VENV/bin/activate"
        echo "[INFO] Persistent paths injected into $VENV/bin/activate"
    fi

    # --- 5. VALIDATION ---
    python - <<'PY'
import torch, vllm, sys
try:
    print(f'vLLM: {vllm.__version__} | GPUs: {torch.cuda.device_count()}')
    if torch.cuda.device_count() == 0:
        print("CRITICAL ERROR: No AMD GPUs detected!")
        sys.exit(1)
except Exception as e:
    print(f"Validation failed: {e}")
    sys.exit(1)
PY
EOF
)

  echo "== Processing $host =="
  if is_local_host "$host"; then
    bash -lc "$install_cmd"
  else
    # Capture the exit code of the remote bash session
    ssh "$target" "bash -lc 'bash -s'" <<EOF || { echo "Setup FAILED on $host"; exit 1; }
$install_cmd
EOF
  fi
}

echo "Using BASE_DIR=$BASE_DIR"
echo "Using VENV=$VENV"
echo "Using RCCL_HOME=$RCCL_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using MPI_COMPAT_HOME=$MPI_COMPAT_HOME"
echo "Using ROCM_PATH=$ROCM_PATH"
echo "Using VLLM_VERSION=$VLLM_VERSION"
echo "Using VLLM index: $VLLM_INDEX_URL"

for host in "${HOSTS[@]}"; do
  echo "== [$(date +%H:%M:%S)] Preflight $host =="
  ensure_rccl_on_host "$host"
  echo " RCCL compatibity tested "
  ensure_mpi_compat_on_host "$host"
  echo " MPI compatibility test "
done

for host in "${HOSTS[@]}"; do
  echo "== [$(date +%H:%M:%S)] Setup vLLM on $host =="
  setup_host "$host"
done

echo "vLLM setup completed on: ${HOSTS[*]}"