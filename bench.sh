#!/usr/bin/env bash
# bench.sh DATA_DIR [RUNS]: best-of-N wall on cpus 0-7, output to tmpfs. Extra env passes through.
set -eu
C="$(cd "$(dirname "$0")" && pwd)"
DATA="$1"; RUNS="${2:-3}"
OUT="${BENCH_OUT:-${TMPDIR:-/tmp}/cand-c/bench}"; mkdir -p "$(dirname "$OUT")"
for i in $(seq "$RUNS"); do
    rm -rf "$OUT"
    s=$(date +%s%N)
    SS_STATS=1 taskset -c 0-7 "$C/target/release/sessions" "$DATA" "$OUT" 2> "$OUT.log"
    e=$(( ($(date +%s%N) - s) / 1000000 ))
    echo "run $i: $e ms  $(grep 'total wall' "$OUT.log")  load $(cut -d' ' -f1 /proc/loadavg)"
done
