#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if command -v python3.12 >/dev/null 2>&1; then
  vtc_python=python3.12
elif command -v python3 >/dev/null 2>&1; then
  vtc_python=python3
else
  echo 'Install 64-bit Python 3.12, then run this script again.' >&2
  exit 2
fi
"$vtc_python" run.py "$@"
