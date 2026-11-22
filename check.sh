#!/usr/bin/env bash
set -euo pipefail
C="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$C/.." && pwd)"
exec "$ROOT/scripts/v5_check.sh" "$C/target/release/sessions" --p20
