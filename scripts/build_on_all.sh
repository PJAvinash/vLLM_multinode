
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../temp" && pwd)}"
MPI_HOME="${MPI_HOME:-${BASE_DIR}/mpich/install}"
HOSTFILE="${HOSTFILE:-${BASE_DIR}/../configs/hostsfile.txt}"

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

${MPI_HOME}/bin/mpirun -np ${n_nodes} -hostfile "${HOSTFILE}"  --bind-to numa "${SCRIPT_DIR}"/build_rccl.sh