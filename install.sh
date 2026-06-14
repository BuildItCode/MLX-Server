#!/usr/bin/env bash
# Install MLX Server Launcher (MLXS) as a global command so you can start it from
# anywhere (like `claude`). Exposes: mlx-launcher, mlxs, mlx-acp-agent.
#
# Tries pipx first (clean isolated global install); if pipx isn't available and
# can't be bootstrapped, falls back to a local venv + symlinks in ~/.local/bin,
# which always works and needs no system-level package installs.
set -uo pipefail   # NB: no -e — a failed bootstrap step must fall through, not abort

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

finish() {
  echo
  echo "Done. Start it from anywhere with:  mlxs"
  echo "(If the command isn't found, open a new terminal so PATH updates take effect.)"
  exit 0
}

# 1) pipx already installed → use it.
if command -v pipx >/dev/null 2>&1; then
  echo "Installing globally with pipx ..."
  if pipx install --force "$HERE"; then
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
    python3 -m pip install --user pipx >/dev/null 2>&1 \
      || python3 -m pip install --user --break-system-packages pipx >/dev/null 2>&1 \
      || true
    python3 -m pipx ensurepath >/dev/null 2>&1 || true
  fi
  hash -r 2>/dev/null || true
  if command -v pipx >/dev/null 2>&1 && pipx install --force "$HERE"; then
    pipx ensurepath >/dev/null 2>&1 || true
    finish
  fi
  echo "Couldn't set up pipx — using a local venv + symlinks instead ..."
fi

# 2) Fallback: local venv + symlinks into ~/.local/bin (self-contained, no system installs).
set -e
echo "Setting up a local venv ..."
VENV="$HERE/.venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -e "$HERE"

BIN="$HOME/.local/bin"
mkdir -p "$BIN"
for cmd in mlx-launcher mlxs mlx-acp-agent; do
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
echo "Done. Start it from anywhere with:  mlxs"
