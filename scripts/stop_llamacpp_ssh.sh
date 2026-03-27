#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Config & Paths
# ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
HOSTFILE="${HOSTFILE:-$BASE_DIR/../configs/hostsfile.txt}"
SSH_USER="${SSH_USER:-$USER}"

# ------------------------
# Host Detection logic
# ------------------------
if [ ! -f "$HOSTFILE" ]; then
    echo "[ERROR] Hostfile not found at $HOSTFILE"
    exit 1
fi

# Get unique hosts from the hostfile (stripping comments and empty lines)
mapfile -t ALL_HOSTS < <(awk '!/^[[:space:]]*($|#)/{print $1}' "$HOSTFILE" | sort -u)

LOCAL_HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
LOCAL_HOST_FULL="$(hostname 2>/dev/null || true)"

is_local_target() {
    local target="$1"
    [[ "$target" == "localhost" || "$target" == "127.0.0.1" || "$target" == "$LOCAL_HOST_SHORT" || "$target" == "$LOCAL_HOST_FULL" ]] && return 0
    return 1
}

ssh_cmd() {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$@"
}

# ------------------------
# Cleanup Command
# ------------------------
# We define the kill logic once to ensure consistency
KILL_CMD="pkill -u $SSH_USER -x llama-server || true; \
          pkill -u $SSH_USER -x rpc-server || true; \
          pkill -u $SSH_USER -f 'rpc-server' || true"

# ------------------------
# Execution Loop
# ------------------------
echo "[INFO] Starting cluster-wide stop sequence..."

for HOST in "${ALL_HOSTS[@]}"; do
    if is_local_target "$HOST"; then
        echo "[LOCAL] Cleaning up processes on $HOST..."
        bash -lc "$KILL_CMD"
    else
        echo "[REMOTE] Cleaning up processes on $HOST..."
        ssh_cmd "${SSH_USER}@${HOST}" "bash -lc '$KILL_CMD'"
    fi
done

echo "[SUCCESS] Cleanup sequence complete."