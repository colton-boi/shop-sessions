#!/usr/bin/env bash
# adversarial.sh BIN [SRC]: rewrite SRC (default data/dev) with tests/adversarial.py, run the
# reference once, then compare BIN at 1/3/4/8 threads, without the pool, without huge pages, with
# forced MADV_HUGEPAGE and at the extent/read-buffer boundaries. Prints ADVERSARIAL PASS or FAIL.
set -uo pipefail
ROOT='/home/colton/Documents/Projects/Builders/#3'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="$(readlink -f "$1")"
SRC="${2:-$ROOT/data/dev}"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
"$ROOT/.venv/bin/python" -I "$HERE/adversarial.py" "$SRC" "$W/data"
"$ROOT/.venv/bin/python" "$ROOT/reference/reference.py" "$W/data" "$W/ref" >/dev/null 2>&1
fail=0
for cfg in "SS_THREADS=1" "SS_THREADS=4" "SS_THREADS=8" "SS_THREADS=8 SS_NOHUGE=1" "SS_THREADS=8 SS_THP=advise" \
    "SS_THREADS=4 SS_EXTENT=192 SS_READ_BUF=4097" "SS_THREADS=4 SS_EXTENT=4097 SS_READ_BUF=192" "SS_THREADS=3 SS_NOPOOL=1"; do
  env $cfg "$B" "$W/data" "$W/o" 2>/dev/null
  r=$("$ROOT/.venv/bin/python" "$ROOT/scripts/compare.py" "$W/o" "$W/ref" | tail -1)
  echo "$cfg: $r"; [ "$r" = MATCH ] || fail=1
  rm -rf "$W/o"
done
echo "ADVERSARIAL $([ $fail = 0 ] && echo PASS || echo FAIL)"
