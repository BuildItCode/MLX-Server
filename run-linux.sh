#!/usr/bin/env bash
# Bootstrap (create venv + install) and launch the launcher on Linux (llama.cpp backend).
#   ./run-linux.sh              # launch (installs on first run)
#   ./run-linux.sh --reinstall  # force a dependency reinstall, then launch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="$HERE/.venv"

pick_python() {
  local c
  for c in "${PYTHON:-}" python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info<(3,15) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

if [ ! -d "$VENV" ]; then
  PYTHON="$(pick_python || true)"
  if [ -z "${PYTHON:-}" ]; then
    echo "No suitable Python found (need 3.10-3.14). Try: sudo apt install python3.12 python3.12-venv" >&2
    exit 1
  fi
  echo "Creating virtual environment in .venv ($("$PYTHON" --version 2>&1)) ..."
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

reinstall=0
if [ "${1:-}" = "--reinstall" ]; then reinstall=1; shift; fi

if [ "$reinstall" -eq 1 ] || ! command -v lis-start >/dev/null 2>&1; then
  echo "Installing dependencies (this runs only when needed) ..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -e "$HERE"
fi

command -v llama-server >/dev/null 2>&1 \
  || echo "NOTE: llama-server not found — run ./install-linux.sh or install llama.cpp first."

exec lis-start "$@"
