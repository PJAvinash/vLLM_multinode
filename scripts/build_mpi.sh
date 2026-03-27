#!/usr/bin/env bash

build_mpi=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"

if [ ${build_mpi} -eq 1 ]
then
    cd ${BASE_DIR}
    if [ ! -d mpich/install ]
    then
        wget https://www.mpich.org/static/downloads/4.1.2/mpich-4.1.2.tar.gz
        mkdir -p mpich
        tar -zxf mpich-4.1.2.tar.gz -C mpich --strip-components=1
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