#!/usr/bin/env bash
# Install the launcher on Linux with the llama.cpp backend.
# MLX is Apple-Silicon-only, so on Linux the usable engine is llama.cpp (`llama-server`).
# This installs the launcher (command: `mlxs`) and best-effort fetches a prebuilt llama-server.
set -uo pipefail   # no -e: a failed optional step should fall through, not abort

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
BIN="$HOME/.local/bin"
mkdir -p "$BIN"

# --- 1) the launcher (pure-Python: pipx if available, else venv + symlinks) ---------
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
  echo "No suitable Python found (need 3.10-3.14)." >&2
  echo "Install one and re-run, e.g.:  sudo apt install python3.12 python3.12-venv" >&2
  exit 1
fi
echo "Using Python: $PYBIN ($("$PYBIN" --version 2>&1))"

installed=0
if command -v pipx >/dev/null 2>&1; then
  echo "Installing the launcher with pipx ..."
  if pipx install --force --python "$PYBIN" "$HERE"; then
    pipx ensurepath >/dev/null 2>&1 || true
    installed=1
  fi
fi
if [ "$installed" -eq 0 ]; then
  echo "Setting up a local venv + symlinks in $BIN ..."
  VENV="$HERE/.venv"
  [ -d "$VENV" ] || "$PYBIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -e "$HERE"
  for cmd in mlx-launcher mlxs mlx-acp-agent; do
    ln -sf "$VENV/bin/$cmd" "$BIN/$cmd"; echo "  linked $BIN/$cmd"
  done
fi

# --- 2) llama.cpp (llama-server) ----------------------------------------------------
if command -v llama-server >/dev/null 2>&1; then
  echo "✓ llama-server already on PATH: $(command -v llama-server)"
else
  echo "Fetching a prebuilt llama-server from github.com/ggml-org/llama.cpp ..."
  case "$(uname -m)" in
    x86_64|amd64)  ASSET="bin-ubuntu-x64.tar.gz" ;;
    aarch64|arm64) ASSET="bin-ubuntu-arm64.tar.gz" ;;
    *)             ASSET="" ;;
  esac
  url=""
  if [ -n "$ASSET" ] && command -v curl >/dev/null 2>&1; then
    url="$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest 2>/dev/null \
            | grep -oE "https://[^\"']*${ASSET}" | head -1)"
  fi
  ok=0
  if [ -n "$url" ]; then
    tmp="$(mktemp -d)"
    dest="$HOME/.local/share/llama.cpp"
    if curl -fsSL "$url" -o "$tmp/llama.tgz" && mkdir -p "$dest" && tar -xzf "$tmp/llama.tgz" -C "$dest"; then
      srv="$(find "$dest" -type f -name llama-server 2>/dev/null | head -1)"
      if [ -n "$srv" ]; then
        chmod +x "$srv" 2>/dev/null || true
        ln -sf "$srv" "$BIN/llama-server"   # shared libs sit beside it; rpath=$ORIGIN finds them
        echo "✓ installed llama-server → $BIN/llama-server (CPU build)"
        ok=1
      fi
    fi
    rm -rf "$tmp"
  fi
  if [ "$ok" -eq 0 ]; then
    echo "Couldn't auto-install llama-server. Install it manually:"
    echo "  • download a build from https://github.com/ggml-org/llama.cpp/releases"
    echo "      ubuntu-x64 = CPU · ubuntu-vulkan/cuda/rocm = GPU"
    echo "  • or build from source: https://github.com/ggml-org/llama.cpp"
    echo "  Put 'llama-server' on your PATH (e.g. symlink it into $BIN)."
  fi
fi

case ":$PATH:" in
  *":$BIN:"*) : ;;
  *)
    echo
    echo "NOTE: $BIN is not on your PATH. Add this to ~/.bashrc or ~/.zshrc, then restart your shell:"
    echo '    export PATH="$HOME/.local/bin:$PATH"' ;;
esac
echo
echo "Done. Start it from anywhere with:  mlxs"
