#!/bin/bash
set -eu

if [[ $# -lt 1 || ${1,,} != *.inp ]]; then
    echo 'Usage: yc A.inp [--scan-hosts lion38 lion39 ...] [scan options]' >&2
    exit 2
fi

input=$1
shift
output=${input%.*}.out
if [[ ! -f "$input" || ! -r "$input" ]]; then
    printf 'YoungCC: cannot read %s\n' "$input" >&2
    exit 2
fi
if [[ "$input" -ef "$output" ]]; then
    printf 'YoungCC: input and output refer to the same file: %s\n' "$input" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
# PySCF owns the OpenMP pool; avoid nested BLAS oversubscription.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
exec "$root/.venv/bin/python" -u -m opendft.interfaces.cli "$@" -- "$input" > "$output" 2>&1
