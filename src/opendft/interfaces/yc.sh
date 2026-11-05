#!/bin/bash
set -eu

if [[ $# != 1 || ${1,,} != *.inp ]]; then
    echo 'Usage: yc A.inp' >&2
    exit 2
fi

input=$1
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
exec "$root/.venv/bin/python" -u -m opendft.interfaces.cli -- "$input" > "$output" 2>&1
