#!/usr/bin/env bash
# Correctness sweep: fixtures and seed 1 with odd reader buffer and extent sizes
# (word, buffer and extent boundaries), then every cached seed and p20 at defaults.
set -euo pipefail
C="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$C/.." && pwd)"
B="$C/target/release/sessions"
cd "$ROOT"
for cfg in "192 1000 1" "192 4097 8" "4096 4097 3" "131072 2097152 8"; do
    set -- $cfg
    export SS_READ_BUF=$1 SS_EXTENT=$2 SS_THREADS=$3
    echo "buf $1 extent $2 threads $3: fixtures $(.venv/bin/python scripts/run_fixtures.py fixtures -- "$B" 2>&1 | tail -1)," \
        "seed 1 $(reference/crosscheck.sh 1 1000000 250000 "$B" 2>/dev/null | tail -1)"
done
unset SS_READ_BUF SS_EXTENT SS_THREADS
for s in "2 1000000 250000" "3 1000000 250000" "4 1000000 250000" "5 1000000 250000" "15 1000000 250000" \
         "6 5000000 1250000" "62 5000000 1250000"; do
    set -- $s
    echo "seed $1: $(reference/crosscheck.sh $1 $2 $3 "$B" 2>/dev/null | tail -1)"
done
rm -rf "$C/tmp/check-dev" "$C/tmp/check-p20"
"$B" data/dev "$C/tmp/check-dev"
echo "dev: $(.venv/bin/python scripts/compare.py "$C/tmp/check-dev" data/dev-answer | tail -1)"
"$B" data/p20 "$C/tmp/check-p20"
echo "p20: $(.venv/bin/python scripts/compare.py "$C/tmp/check-p20" reference/tmp/p20-ref-out | tail -1)"
rm -rf "$C/tmp/check-dev" "$C/tmp/check-p20"
