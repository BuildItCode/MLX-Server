#!/usr/bin/env bash
# Bootstrap (create venv + install) and launch the MLX Server Launcher TUI.
#   ./run.sh              # launch (installs on first run)
#   ./run.sh --reinstall  # force a dependency reinstall, then launch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

reinstall=0
if [ "${1:-}" = "--reinstall" ]; then
  reinstall=1
  shift
fi

if [ "$reinstall" -eq 1 ] || ! command -v mlx-launcher >/dev/null 2>&1; then
  echo "Installing dependencies (this runs only when needed) ..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -e "$HERE"
fi

exec mlx-launcher "$@"
