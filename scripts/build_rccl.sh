#!/usr/bin/env bash
set -euo pipefail
build_rccl=1
rccl_debug_mode=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
RCCL_INSTALL_DIR=""

## building rccl
if [ ${build_rccl} -eq 1 ]
then
    RCCL_BUILD_TYPE=release
    cd ${BASE_DIR}
    if [ ! -d rocm-systems/projects/rccl ]
    then
        git clone https://github.com/ROCm/rocm-systems.git
    fi
    cd ${BASE_DIR}/rocm-systems/projects/rccl
    mkdir -p ${BASE_DIR}/logs
    if [ ${rccl_debug_mode} -eq 1 ]
    then
        ./install.sh -l --debug --jobs $(nproc) > ${BASE_DIR}/logs/rccl_build.log 2>&1
        RCCL_BUILD_TYPE=debug
        echo "Building rccl in debug mode"
    else
        ./install.sh -l --jobs $(nproc) > ${BASE_DIR}/logs/rccl_build.log 2>&1
        RCCL_BUILD_TYPE=release
    fi
    RCCL_INSTALL_DIR=${BASE_DIR}/rocm-systems/projects/rccl/build/${RCCL_BUILD_TYPE}
    echo "[INFO] RCCL build(home) is at ${RCCL_INSTALL_DIR}"
fi