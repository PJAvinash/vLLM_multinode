#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Config
# ------------------------
SSH_USER="${SSH_USER:-$USER}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/../configs/hostsfile.txt}"

VENV="${VENV:-$BASE_DIR/.venv}"
RCCL_HOME="${RCCL_HOME:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
ROCM_PATH="${ROCM_PATH:-$(readlink -f /opt/rocm/core 2>/dev/null || true)}"
MPI_HOME="${MPI_HOME:-$BASE_DIR/openmpi}"

# ------------------------
# Detect local host
# ------------------------
LOCAL_HOST_FULL="$(hostname)"
LOCAL_HOST_SHORT="$(hostname -s)"
LOCAL_HOST_FQDN="$(hostname -f 2>/dev/null || true)"
mapfile -t LOCAL_IPS < <(hostname -I | tr ' ' '\n' | sed '/^$/d')

# ------------------------
# Parse hostfile
# ------------------------
mapfile -t HOSTS < <(awk '!/^[[:space:]]*($|#)/{print $1}' "$HOSTFILE")
[ "${#HOSTS[@]}" -gt 0 ] || { echo "[ERROR] Hostfile empty"; exit 1; }

# ------------------------
# Helpers
# ------------------------
is_self() {
    local h="$1"
    [[ "$h" == "localhost" || "$h" == "127.0.0.1" ]] && return 0
    [[ "$h" == "$LOCAL_HOST_FULL" || "$h" == "$LOCAL_HOST_SHORT" || "$h" == "$LOCAL_HOST_FQDN" ]] && return 0
    for ip in "${LOCAL_IPS[@]}"; do [[ "$h" == "$ip" ]] && return 0; done
    return 1
}

ssh_cmd() {
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=20 \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=6 \
    "$@"
}

run_local_stop() {
  set +u
  source /etc/profile >/dev/null 2>&1 || true
  set -u

  if [ -d "$VENV" ]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
  fi

  if [ -n "$ROCM_PATH" ] && [ -d "$ROCM_PATH/lib" ]; then
    export LD_LIBRARY_PATH="$MPI_HOME/lib:$RCCL_HOME:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="$MPI_HOME/lib:$RCCL_HOME:${LD_LIBRARY_PATH:-}"
  fi

  pkill -f "vllm.*serve" || true
  ray stop --force || true
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

if [ -z "$MASTER_HOST" ]; then
    MASTER_HOST="${HOSTS[0]}"
    WORKER_HOSTS=("${HOSTS[@]:1}")
    echo "[WARN] Local host not in hostfile, using $MASTER_HOST as master"
fi

echo "[INFO] Master: $MASTER_HOST"
echo "[INFO] Workers: ${WORKER_HOSTS[*]}"

# ------------------------
# Stop processes
# ------------------------
for H in "$MASTER_HOST" "${WORKER_HOSTS[@]}"; do
  echo "[INFO] Stopping services on $H"

  if is_self "$H"; then
    run_local_stop
  else
    ssh_cmd "${SSH_USER}@${H}" "bash -lc '
      set +u
      source /etc/profile >/dev/null 2>&1 || true
      set -u
      if [ -d \"$VENV\" ]; then source \"$VENV/bin/activate\"; fi
      if [ -n \"$ROCM_PATH\" ] && [ -d \"$ROCM_PATH/lib\" ]; then
        export LD_LIBRARY_PATH=\"$MPI_HOME/lib:$RCCL_HOME:$ROCM_PATH/lib:\${LD_LIBRARY_PATH:-}\"
      else
        export LD_LIBRARY_PATH=\"$MPI_HOME/lib:$RCCL_HOME:\${LD_LIBRARY_PATH:-}\"
      fi
      pkill -f \"vllm.*serve\" || true
      ray stop --force || true
    '" || echo "[WARN] Failed on $H"
  fi
done