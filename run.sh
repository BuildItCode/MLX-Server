#!/usr/bin/env bash
# Bootstrap (create venv + install) and launch the LIS (Local Inference Server) TUI.
#   ./run.sh              # launch (installs on first run)
#   ./run.sh --reinstall  # force a dependency reinstall, then launch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"

# Pick a Python that satisfies requires-python (>=3.10,<3.15). macOS's
# /usr/bin/python3 is usually 3.9, which the launcher rejects — so don't trust
# bare `python3`. Honor an explicit $PYTHON override if the user set one.
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

# Recreate the venv if it's missing, broken, or MOVED. A bare `.venv` dir isn't enough:
#  - if the base Python it was built against was upgraded/removed, `.venv/bin/python` won't run; and
#  - venvs aren't relocatable: if `.venv` was created in another folder (e.g. the zip's
#    `MLX-Server-main/`) and then renamed/copied here, its `activate` + console-script shebangs still
#    point at the OLD absolute path, so `python`/`lis-start` resolve to nothing on PATH.
# `activate` always names its own VIRTUAL_ENV dir, so if it no longer mentions THIS `.venv`, it moved.
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
    echo "No suitable Python found (need 3.10–3.14). Try: brew install python@3.12" >&2
    exit 1
  fi
  echo "Creating virtual environment in .venv ($("$PYTHON" --version 2>&1)) ..."
  "$PYTHON" -m venv "$VENV"
fi

# Use the venv's interpreter/scripts by explicit path — don't trust a bare `python`/`lis-start` on
# PATH (activation can be a no-op in odd shells, and a global lis-* could mask the venv's).
VPY="$VENV/bin/python"
# shellcheck disable=SC1091
source "$VENV/bin/activate"  # still set VIRTUAL_ENV/PATH for the launched process + its children

reinstall=0
if [ "${1:-}" = "--reinstall" ]; then
  reinstall=1
  shift
fi

# Gate on the venv's own lis-backend (the newer entry point): a missing one means a fresh venv or an
# install predating the backend split, so (re)install to pick up its new deps (starlette/uvicorn/…).
# Check the file directly, not PATH, so a global lis-backend can't mask a venv that still needs it.
if [ "$reinstall" -eq 1 ] || [ ! -x "$VENV/bin/lis-backend" ]; then
  echo "Installing dependencies (this runs only when needed) ..."
  "$VPY" -m pip install --quiet --upgrade pip
  "$VPY" -m pip install --quiet -e "$HERE"
fi

exec "$VENV/bin/lis-start" "$@"
