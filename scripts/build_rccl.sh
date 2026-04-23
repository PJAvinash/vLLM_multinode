#!/usr/bin/env bash
set -euo pipefail

build_rccl=${build_rccl:-1}
rccl_debug_mode=${rccl_debug_mode:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
RCCL_INSTALL_DIR=""

# Allow override via environment variables
BRANCH_NAME="${BRANCH_NAME:-}"   # optional , but default is set as develop
COMMIT_HASH="${COMMIT_HASH:-}"   # optional

echo "[INFO] Using branch: ${BRANCH_NAME}"
if [ -n "${COMMIT_HASH}" ]; then
  echo "[INFO] Using commit: ${COMMIT_HASH}"
fi

## building rccl
if [ "${build_rccl}" -eq 1 ]; then
    RCCL_BUILD_TYPE=release
    cd "${BASE_DIR}"

    if [ ! -d rocm-systems/projects/rccl ]; then
        git clone https://github.com/ROCm/rocm-systems.git
    fi

    cd "${BASE_DIR}/rocm-systems/projects/rccl"

    git fetch --all

    # Checkout branch first
    # If BRANCH_NAME is provided, override branch
    if [ -n "${BRANCH_NAME}" ]; then
        git checkout "${BRANCH_NAME}"
        git pull
    fi

    # If commit hash is provided, override branch
    if [ -n "${COMMIT_HASH}" ]; then
        git checkout "${COMMIT_HASH}"
    fi

    mkdir -p "${BASE_DIR}/logs"

    if [ "${rccl_debug_mode}" -eq 1 ]; then
        echo "[INFO] Building RCCL in DEBUG mode"
        ./install.sh -l --debug --jobs "$(nproc)" \
          > "${BASE_DIR}/logs/rccl_build.log" 2>&1
        RCCL_BUILD_TYPE=debug
    else
        echo "[INFO] Building RCCL in RELEASE mode"
        ./install.sh -l --jobs "$(nproc)" \
          > "${BASE_DIR}/logs/rccl_build.log" 2>&1
        RCCL_BUILD_TYPE=release
    fi

    RCCL_INSTALL_DIR="${BASE_DIR}/rocm-systems/projects/rccl/build/${RCCL_BUILD_TYPE}"
    echo "[INFO] RCCL build (home) is at ${RCCL_INSTALL_DIR}"
fi