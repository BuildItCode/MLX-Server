#!/usr/bin/env bash
# Install MLX Server Launcher as a global command so you can start it from anywhere
# (like `claude`). Exposes: mlx-launcher, mlxs, mlx-acp-agent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if command -v pipx >/dev/null 2>&1; then
  echo "Installing globally with pipx ..."
  pipx install --force "$HERE"
  echo
  echo "Done. Start it from anywhere with:  mlxs"
  exit 0
fi

echo "pipx is not installed."
read -r -p "Install pipx now via 'python3 -m pip install --user pipx'? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  python3 -m pip install --user pipx
  python3 -m pipx ensurepath || true
  echo
  echo "pipx installed. Open a new shell, then re-run ./install.sh"
  exit 0
fi

# Fallback: local venv + symlinks into ~/.local/bin
echo "Falling back to a local venv + symlinks in ~/.local/bin ..."
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
    echo "WARNING: $BIN is not on your PATH. Add this line to ~/.zshrc and restart your shell:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac
echo
echo "Done. Start it from anywhere with:  mlxs"
