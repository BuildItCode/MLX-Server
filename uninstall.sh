#!/usr/bin/env bash
# Uninstall LIS (Local Inference Server) — macOS + Linux.
# Removes the global command(s): the OLD names (mlxs / mlx-launcher) AND the new one
# (lis-start), the pipx install, and this repo's local .venv. Your config (server profiles
# + chats, at ~/.config/mlx-launcher) is KEPT unless you pass --purge.
#
#   ./uninstall.sh            # remove the app, keep your profiles + chats
#   ./uninstall.sh --purge    # also delete the config (profiles + chats)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

removed_any=0

# 1) pipx install (the distribution name is 'mlx-launcher' for both the old + new command
#    sets). Idempotent — ignore if it isn't installed that way.
if command -v pipx >/dev/null 2>&1; then
  if pipx uninstall mlx-launcher >/dev/null 2>&1; then
    echo "✓ removed the pipx install"; removed_any=1
  fi
fi

# 2) ~/.local/bin commands — both the old and the new names (created by install.sh's
#    venv-fallback path with `ln -sf`).
BIN="$HOME/.local/bin"
for cmd in lis-start mlxs mlx-launcher mlx-acp-agent; do
  if [ -e "$BIN/$cmd" ] || [ -L "$BIN/$cmd" ]; then
    rm -f "$BIN/$cmd" && { echo "✓ removed $BIN/$cmd"; removed_any=1; }
  fi
done

# 3) this repo's local dev/run .venv (recreated on the next ./run.sh if you keep the repo).
if [ -d "$HERE/.venv" ]; then
  rm -rf "$HERE/.venv" && { echo "✓ removed $HERE/.venv"; removed_any=1; }
fi

# 4) config / user data.
CFG="${XDG_CONFIG_HOME:-$HOME/.config}/mlx-launcher"
if [ "$PURGE" -eq 1 ]; then
  [ -d "$CFG" ] && rm -rf "$CFG" && echo "✓ removed $CFG (profiles + chats)"
elif [ -d "$CFG" ]; then
  echo "• kept your config at $CFG  (re-run with --purge to delete profiles + chats)"
fi

[ "$removed_any" -eq 0 ] && echo "Nothing to remove — LIS doesn't appear to be installed."
echo
echo "Done. (llama.cpp / the MLX engines are left installed — remove those separately if you want.)"
