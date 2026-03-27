#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Config / Inputs
# ------------------------
HOSTFILE="${HOSTFILE:-${1:-hostsfile.txt}}"
SSH_USER="${SSH_USER:-$USER}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_rsa}"
PUB_KEY="${KEY_PATH}.pub"

if [ ! -f "$HOSTFILE" ]; then
    echo "[ERROR] Hostfile not found: $HOSTFILE"
    exit 1
fi

echo "[INFO] Using hostfile: $HOSTFILE"
echo "[INFO] Using SSH user: $SSH_USER"
echo "[INFO] Key path: $KEY_PATH"

# ------------------------
# 1. Generate RSA key if not exist
# ------------------------
if [ ! -f "$KEY_PATH" ]; then
    echo "[INFO] Generating RSA key..."
    ssh-keygen -t rsa -b 4096 -C "cluster-key" -f "$KEY_PATH" -N ""
else
    echo "[INFO] RSA key already exists"
fi

# Show SHA256 fingerprint
echo "[INFO] Key fingerprint:"
ssh-keygen -lf "$PUB_KEY"

# ------------------------
# 2. Parse hostfile
# ------------------------
mapfile -t HOSTS < <(awk '!/^[[:space:]]*($|#)/{print $1}' "$HOSTFILE")
if [ "${#HOSTS[@]}" -eq 0 ]; then
    echo "[ERROR] Hostfile is empty"
    exit 1
fi

# ------------------------
# 3. Detect master (self)
# ------------------------
LOCAL_HOST_FULL="$(hostname)"
LOCAL_HOST_SHORT="$(hostname -s)"
mapfile -t LOCAL_IPS < <(hostname -I | tr ' ' '\n' | sed '/^$/d')

is_self() {
    local h="$1"
    [[ "$h" == "localhost" || "$h" == "127.0.0.1" ]] && return 0
    [[ "$h" == "$LOCAL_HOST_FULL" || "$h" == "$LOCAL_HOST_SHORT" ]] && return 0
    for ip in "${LOCAL_IPS[@]}"; do
        [[ "$h" == "$ip" ]] && return 0
    done
    return 1
}

echo "[INFO] Local host detected as: $LOCAL_HOST_FULL"

# ------------------------
# 4. SSH config (avoid GSSAPI delays)
# ------------------------
SSH_CONFIG="$HOME/.ssh/config"

if ! grep -q "cluster-auto" "$SSH_CONFIG" 2>/dev/null; then
    echo "[INFO] Updating SSH config..."
    cat >> "$SSH_CONFIG" <<EOF

# cluster-auto
Host *
    GSSAPIAuthentication no
    PreferredAuthentications publickey
    StrictHostKeyChecking accept-new
EOF
fi

# ------------------------
# 5. Copy key to all remote nodes
# ------------------------
for host in "${HOSTS[@]}"; do
    if is_self "$host"; then
        echo "[INFO] Skipping self: $host"
        continue
    fi

    echo "[INFO] Attempting to authorize $host..."

    # Use a temporary SSH option to ensure we can use a password if the key fails
    SSH_OPTS="-o PreferredAuthentications=password,publickey -o StrictHostKeyChecking=accept-new"

    if command -v ssh-copy-id >/dev/null 2>&1; then
        # ssh-copy-id will check if the key exists for you
        ssh-copy-id $SSH_OPTS -i "$PUB_KEY" "${SSH_USER}@${host}"
    else
        echo "[INFO] ssh-copy-id not available, using manual method"
        cat "$PUB_KEY" | ssh $SSH_OPTS "${SSH_USER}@${host}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    fi
done

# ------------------------
# 6. Validate passwordless SSH
# ------------------------
echo "[INFO] Validating SSH access..."

FAIL=0
for host in "${HOSTS[@]}"; do
    if is_self "$host"; then
        continue
    fi

    if ssh -o BatchMode=yes "${SSH_USER}@${host}" "hostname" >/dev/null 2>&1; then
        echo "[OK] $host"
    else
        echo "[FAIL] $host"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 0 ]; then
    echo "[SUCCESS] Passwordless SSH setup complete!"
else
    echo "[ERROR] Some hosts failed. Check SSH manually."
    exit 1
fi