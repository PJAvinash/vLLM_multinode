#!/usr/bin/env bash
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"

MPI_HOME="${MPI_HOME:-${BASE_DIR}/mpich/install}"
RCCL_INSTALL_DIR="${RCCL_INSTALL_DIR:-$BASE_DIR/rocm-systems/projects/rccl/build/release}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-"${ROCM_PATH}/lib"}"
N_GPUS="${N_GPUS:-1}"
HOSTFILE="${HOSTFILE:-${BASE_DIR}/../configs/hostsfile.txt}"

mpi_mode="${MPI_MODE:-1}"
multinode_mode="${MULTINODE_MODE:-0}"
rebuild_rccl_tests="${REBUILD_RT:-0}"

# Always required
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found"; exit 1; }
[ -d "${ROCM_PATH}" ] || { echo "ERROR: ROCm not found at ${ROCM_PATH}"; exit 1; }
[ -d "${RCCL_INSTALL_DIR}" ] || { echo "ERROR: RCCL not found at ${RCCL_INSTALL_DIR}"; exit 1; }

# Only required if MPI mode is enabled
if [[ "${MPI_MODE:-1}" -eq 1 ]]; then
  [ -d "${MPI_HOME}" ] || { echo "ERROR: MPI not found at ${MPI_HOME}"; exit 1; }
  # Also verify mpirun exists (more important than just directory)
  command -v "${MPI_HOME}/bin/mpirun" >/dev/null 2>&1 || {
    echo "ERROR: mpirun not found in ${MPI_HOME}/bin"
    exit 1
  }

  if [[ "${multinode_mode}" -eq 1 ]]; then
    [ -f "${HOSTFILE}" ] || { echo "ERROR: Hostfile not found at ${HOSTFILE}"; exit 1; }
  fi
fi

cd "${BASE_DIR}"
if [ ! -d "${BASE_DIR}/rocm-systems/projects/rccl-tests" ]
then
  git clone https://github.com/ROCm/rocm-systems.git
fi

## building rccl-tests (develop)
BUILD_DIR="${BASE_DIR}/rocm-systems/projects/rccl-tests"
cd "${BASE_DIR}"
if [ "${rebuild_rccl_tests}" -eq 1 ] || [ ! -d "${BUILD_DIR}/build" ]; then
  cd "${BUILD_DIR}"
  make clean
  make MPI="${mpi_mode}" MPI_HOME="${MPI_HOME}" NCCL_HOME="${RCCL_INSTALL_DIR}" -j
fi

## running rccl-tests sweep
n_gpus="${N_GPUS:-$(rocminfo | grep -c 'Device Type: GPU' || echo 1)}" 
n_nodes=1

if [[ "${multinode_mode:-0}" -eq 1 ]]; then
  HOSTFILE="${HOSTFILE:-${BASE_DIR}/hostsfile.txt}"

  if [[ ! -f "${HOSTFILE}" ]]; then
    echo "ERROR: Hostfile not found at ${HOSTFILE}"
    exit 1
  fi

  # Extract hostnames:
  # - remove comments
  # - remove empty lines
  # - strip slots (= or :)
  # - count unique hosts
  n_nodes=$(grep -vE '^\s*#' "${HOSTFILE}" | \
            awk '{print $1}' | \
            sed 's/:.*//' | \
            sort -u | \
            wc -l)

fi

total=$((n_gpus * n_nodes))     # total number of MPI ranks Assumes homogenius cluster
echo "Total ranks: ${total}"    # print number of GPUs
cd ${BASE_DIR}
# Get today's UTC date in yyyy_MM_dd format
DATE_UTC=$(date -u +"%Y_%m_%d")
# Set performance data directory name
PERF_DATA_DIR="perfdata_${DATE_UTC}"
mkdir -p ${PERF_DATA_DIR}

#Run parameters
b=1        #begin size
e=16G      #end size
d=float     #data types
n=1        #iterations
w=0        #warm up iterations
N=1        #stress cycle iterations
 
for coll in all_reduce all_gather alltoall alltoallv broadcast gather reduce reduce_scatter scatter sendrecv
do
    # using MPICH; comment next line if using OMPI
    if [[ $mpi_mode -eq 1 ]]; then
        if [[ $multinode_mode -eq 1 ]]; then 
            echo "[INFO] Running in multinode mode"
            ${MPI_HOME}/bin/mpirun -np ${total} -hostfile "${HOSTFILE}" --bind-to numa -env NCCL_DEBUG=VERSION -env PATH=${MPI_HOME}/bin:${ROCM_PATH}/bin:$PATH -env LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}:${MPI_HOME}/lib:$LD_LIBRARY_PATH ${BASE_DIR}/rocm-systems/projects/rccl-tests/build/${coll}_perf -b ${b} -e ${e} -f 2 -g 1 -d ${d} -n ${n} -w ${w} -N ${N} -M 1 2>&1 | tee ${BASE_DIR}/${PERF_DATA_DIR}/${coll}.txt
        else
            ${MPI_HOME}/bin/mpirun -np ${total} --bind-to numa -env NCCL_DEBUG=VERSION -env PATH=${MPI_HOME}/bin:${ROCM_PATH}/bin:$PATH -env LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}:${MPI_HOME}/lib:$LD_LIBRARY_PATH ${BASE_DIR}/rocm-systems/projects/rccl-tests/build/${coll}_perf -b ${b} -e ${e} -f 2 -g 1 -d ${d} -n ${n} -w ${w} -N ${N} -M 1 2>&1 | tee ${BASE_DIR}/${PERF_DATA_DIR}/${coll}.txt
        fi 
    else 
         NCCL_DEBUG=VERSION PATH=${ROCM_PATH}/bin:$PATH LD_LIBRARY_PATH=${RCCL_INSTALL_DIR}:$LD_LIBRARY_PATH ${BASE_DIR}/rocm-systems/projects/rccl-tests/build/${coll}_perf -b ${b} -e ${e} -f 2 -g ${n_gpus} -d ${d} -n ${n} -w ${w} -N ${N} 2>&1 | tee ${BASE_DIR}/${PERF_DATA_DIR}/${coll}.txt
    fi 
done