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

# Recreate the venv if missing, broken, or MOVED. If the base Python was upgraded/removed,
# .venv/bin/python won't run; and venvs aren't relocatable — if .venv was created elsewhere and
# renamed/copied here, its activate + console-script shebangs point at the old path (so python /
# lis-start vanish). activate always names its own dir, so if it no longer mentions THIS .venv, it moved.
recreate=""
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c '' >/dev/null 2>&1; then
  recreate="its Python won't run"
elif [ -f "$VENV/bin/activate" ] && ! grep -qF "$VENV" "$VENV/bin/activate"; then
  recreate="it was created in a different folder and moved here"
fi
if [ -n "$recreate" ] || [ ! -d "$VENV" ]; then
  if [ -d "$VENV" ]; then
    echo "Recreating .venv ($recreate) ..."
    rm -rf "$VENV"
  fi
  PYTHON="$(pick_python || true)"
  if [ -z "${PYTHON:-}" ]; then
    echo "No suitable Python found (need 3.10-3.14). Try: sudo apt install python3.12 python3.12-venv" >&2
    exit 1
  fi
  echo "Creating virtual environment in .venv ($("$PYTHON" --version 2>&1)) ..."
  "$PYTHON" -m venv "$VENV"
fi

# Use the venv's interpreter/scripts by explicit path — don't trust a bare `python`/`lis-start`.
VPY="$VENV/bin/python"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

reinstall=0
if [ "${1:-}" = "--reinstall" ]; then reinstall=1; shift; fi

# Gate on the venv's own lis-backend (the newer entry point); check the file, not PATH, so a global
# lis-backend can't mask a venv that still needs the new deps (starlette/uvicorn/…).
if [ "$reinstall" -eq 1 ] || [ ! -x "$VENV/bin/lis-backend" ]; then
  echo "Installing dependencies (this runs only when needed) ..."
  "$VPY" -m pip install --quiet --upgrade pip
  "$VPY" -m pip install --quiet -e "$HERE"
fi

command -v llama-server >/dev/null 2>&1 \
  || echo "NOTE: llama-server not found — run ./install-linux.sh or install llama.cpp first."

exec "$VENV/bin/lis-start" "$@"
