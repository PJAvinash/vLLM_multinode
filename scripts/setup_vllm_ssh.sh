#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/../configs/hostsfile.txt}"
VENV="${VENV:-$BASE_DIR/.venv}"
RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

MPI_HOME="${MPI_HOME:-${BASE_DIR}/mpich/install}"
MPI_COMPAT_HOME="${MPI_COMPAT_HOME:-/usr}"
VLLM_VERSION="${VLLM_VERSION:-0.17.0+rocm700}"
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
  
  

  local install_cmd=$(cat <<EOF
    set -euo pipefail
    # Ensure these paths are consistent across the cluster
    CONDA_DIR="\$HOME/miniconda3"
    CONDA_BIN="\$CONDA_DIR/bin/conda"
    mkdir -p "$BASE_DIR"
    cd "$BASE_DIR"

    # --- 1. FIND OR INSTALL COMPATIBLE PYTHON ---
    PYTHON_EXE=""
    # Check for existing compatible system python (3.9 - 3.12)
    for cmd in python3.12 python3.11 python3.10 python3; do
        if command -v \$cmd >/dev/null 2>&1; then
            IS_COMPATIBLE=\$(\$cmd -c 'import sys; print(int(sys.version_info < (3, 13) and sys.version_info >= (3, 9)))')
            if [ "\$IS_COMPATIBLE" -eq "1" ]; then
                PYTHON_EXE=\$(command -v \$cmd)
                break
            fi
        fi
    done

    # FALLBACK: Install Miniconda if no compatible Python found
    if [ -z "\$PYTHON_EXE" ]; then
        echo "[INFO] No compatible system Python (3.9-3.12) found. Deploying Miniconda..."
        if [ ! -f "$CONDA_BIN" ]; then
            curl -L -s https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
            bash miniconda.sh -b -p "$CONDA_DIR"
            rm miniconda.sh
        fi
        
        # Initialize conda for this session
        source "$CONDA_DIR/etc/profile.d/conda.sh"
        
        # Create/Activate environment
        if ! conda info --envs | grep -q "vllm_env"; then
            conda create -y -n vllm_env python=3.12
        fi
        conda activate vllm_env
        PYTHON_EXE=\$(which python)
    else
        # Standard venv path if system python was found
        echo "[INFO] Using system Python: \$PYTHON_EXE"
        "\$PYTHON_EXE" -m venv "$VENV" --system-site-packages || "\$PYTHON_EXE" -m venv "$VENV"
        source "$VENV/bin/activate"
        PYTHON_EXE=\$(which python)
    fi

    # --- 2. INSTALLATION ---
    echo "[INFO] Using Python: \$PYTHON_EXE"
    "\$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel

    export HSAKMT_DEBUG_LEVEL=0
    export ROCM_DISABLE_LLC=1

    PIP_FLAGS=""
    [ "\${DEBUG_INSTALL:-0}" = "1" ] && PIP_FLAGS="-v"

    # Install PyTorch ROCm 6.0
    "\$PYTHON_EXE" -m pip install \$PIP_FLAGS \
      --only-binary=:all: \
      torch torchvision \
      --index-url https://download.pytorch.org/whl/rocm6.0

    # Install vLLM
    "\$PYTHON_EXE" -m pip install \$PIP_FLAGS \
      --only-binary=:all: \
      "vllm==$VLLM_VERSION" \
      --extra-index-url "$VLLM_INDEX_URL"

    # --- 3. LIBRARY PATHS ---
    MPI_COMPAT_LIB="$MPI_COMPAT_HOME/lib"
    if [ -d "$ROCM_PATH/lib" ]; then
      export LD_LIBRARY_PATH="\${MPI_COMPAT_LIB}:$MPI_HOME/lib:$RCCL_HOME:$ROCM_PATH/lib:\${LD_LIBRARY_PATH:-}"
    else
      export LD_LIBRARY_PATH="\${MPI_COMPAT_LIB}:$MPI_HOME/lib:$RCCL_HOME:\${LD_LIBRARY_PATH:-}"
    fi

    # --- 4. VALIDATION ---
    "\$PYTHON_EXE" - <<'PY'
import torch, vllm, sys
print(f'Target: {sys.executable}')
print(f'vLLM: {vllm.__version__} | Torch: {torch.__version__}')
print(f'HIP: {getattr(torch.version, "hip", "None")} | GPUs: {torch.cuda.device_count()}')
PY
EOF
)

  echo "== Processing $host =="
  if is_local_host "$host"; then
    bash -lc "$install_cmd"
  else
    # Pipe the command via stdin to avoid SSH argument length limits
    ssh "$target" "bash -lc 'bash -s'" <<EOF
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