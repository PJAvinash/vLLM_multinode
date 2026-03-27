#!/usr/bin/env bash
set -euo pipefail
build_mpi=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/../temp"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"

MPICH_VERSION=4.1.2

if [ ${build_mpi} -eq 1 ]
then
    cd ${BASE_DIR}
    if [ ! -d mpich/install ]
    then
        wget "https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz"
        mkdir -p mpich
        tar -zxf "mpich-${MPICH_VERSION}.tar.gz" -C mpich --strip-components=1
        cd mpich
        mkdir build
        cd build
        ../configure --prefix=${BASE_DIR}/mpich/install --disable-fortran --with-ucx=embedded
        make -j 16
        make install
    fi
    MPI_INSTALL_DIR=${BASE_DIR}/mpich/install
    echo "[INFO] MPI build(home) is at ${MPI_INSTALL_DIR}"
fi