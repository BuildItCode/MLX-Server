#!/usr/bin/env bash
# Install LIS (Local Inference Server) as a global command so you can start it from
# anywhere (like `claude`). Exposes: lis-start, mlx-acp-agent.
#
# Tries pipx first (clean isolated global install); if pipx isn't available and
# can't be bootstrapped, falls back to a local venv + symlinks in ~/.local/bin,
# which always works and needs no system-level package installs.
set -uo pipefail   # NB: no -e — a failed bootstrap step must fall through, not abort

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pick a Python that satisfies the launcher's requires-python (>=3.10,<3.15).
# macOS's /usr/bin/python3 is usually 3.9 — too old for the launcher, and it also
# forces ancient mlx-lm/mlx-vlm (the old mlx-vlm has no mlx_vlm.server at all). So
# search for a modern interpreter explicitly instead of trusting bare `python3`.
pick_python() {
  local c
  for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info<(3,15) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

PYBIN="$(pick_python || true)"
if [ -z "${PYBIN:-}" ]; then
  echo "No suitable Python found (need 3.10–3.14)."
  echo "Install one, then re-run ./install.sh, e.g.:"
  echo "    brew install python@3.12        # Homebrew"
  echo "    uv python install 3.12          # if you use uv"
  exit 1
fi
echo "Using Python: $PYBIN ($("$PYBIN" --version 2>&1))"

finish() {
  echo
  echo "Done. Start it from anywhere with:  lis-start"
  echo "(If the command isn't found, open a new terminal so PATH updates take effect.)"
  exit 0
}

# 1) pipx already installed → use it. Pin the modern interpreter so pipx doesn't
#    build the launcher's venv with an old system Python.
if command -v pipx >/dev/null 2>&1; then
  echo "Installing globally with pipx ..."
  if pipx install --force --python "$PYBIN" "$HERE"; then
    pipx ensurepath >/dev/null 2>&1 || true
    finish
  fi
  echo "pipx install failed — falling back to a local venv ..."
else
  echo "pipx is not installed."
  # On Homebrew's Python, 'pip install' is blocked (PEP 668 externally-managed),
  # so prefer 'brew install pipx'; otherwise try pip with a --break-system-packages
  # fallback. Every step tolerates failure and falls through to the venv path.
  if command -v brew >/dev/null 2>&1; then
    read -r -p "Install pipx via 'brew install pipx'? [Y/n] " ans
    [[ "${ans:-Y}" =~ ^[Nn]$ ]] || brew install pipx || true
  fi
  if ! command -v pipx >/dev/null 2>&1; then
    "$PYBIN" -m pip install --user pipx >/dev/null 2>&1 \
      || "$PYBIN" -m pip install --user --break-system-packages pipx >/dev/null 2>&1 \
      || true
    "$PYBIN" -m pipx ensurepath >/dev/null 2>&1 || true
  fi
  hash -r 2>/dev/null || true
  if command -v pipx >/dev/null 2>&1 && pipx install --force --python "$PYBIN" "$HERE"; then
    pipx ensurepath >/dev/null 2>&1 || true
    finish
  fi
  echo "Couldn't set up pipx — using a local venv + symlinks instead ..."
fi

# 2) Fallback: local venv + symlinks into ~/.local/bin (self-contained, no system installs).
set -e
echo "Setting up a local venv ..."
VENV="$HERE/.venv"
[ -d "$VENV" ] || "$PYBIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -e "$HERE"

BIN="$HOME/.local/bin"
mkdir -p "$BIN"
for cmd in lis-start mlx-acp-agent; do
  ln -sf "$VENV/bin/$cmd" "$BIN/$cmd"
  echo "  linked $BIN/$cmd"
done

case ":$PATH:" in
  *":$BIN:"*) : ;;
  *)
    echo
    echo "NOTE: $BIN is not on your PATH. Add this to ~/.zshrc, then restart your shell:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac
echo
echo "Done. Start it from anywhere with:  lis-start"
