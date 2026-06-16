# AGENTS.md — working on MLX Server Launcher

Guidance for AI coding agents and human contributors working on **this repository**. (Not to
be confused with the per-project `AGENTS.md` the app itself reads when you point a chat at a
folder as a coding workspace.)

## What this is

A Claude-Code-styled **terminal UI (TUI)** that launches and manages local model servers, chats
with them, and wires them into Xcode 27. It is a thin, local-first **orchestrator** — it does
not run inference itself. It spawns an OpenAI-compatible **model server** as a subprocess and
talks HTTP to it.

- **Stack:** Python 3.10–3.14, [Textual](https://textual.textualize.io) (TUI), httpx,
  pydantic v2, and the `mcp`, `ddgs`, `pypdf`, and `agent-client-protocol` (`acp`) libraries.
- **Entry points** (`pyproject.toml` `[project.scripts]`): `mlxs` / `mlx-launcher`
  (`mlx_launcher.app:run`, the TUI) and `mlx-acp-agent` (`mlx_launcher.acp.entry:main`, the
  stdio ACP agent Xcode launches).
- **Pure-Python deps only.** The model runtimes (mlx-lm, mlx-vlm, vllm-mlx, llama.cpp) are
  **external binaries** installed separately and resolved on `PATH` — they are never imported.

## Engines

Four interchangeable backends, chosen per server profile via `ServerConfig.engine`
(`config/models.py`). All are OpenAI-compatible over HTTP, so the chat / bridge / ACP layers
are engine-agnostic.

| engine | binary | models | platform |
|---|---|---|---|
| `mlx-lm` | `mlx_lm.server` | text LLMs | Apple Silicon |
| `mlx-vlm` | `mlx_vlm.server` | vision-language | Apple Silicon |
| `vllm-mlx` | `vllm-mlx serve` | text + vision, KV-quant, native tools | Apple Silicon |
| `llama-cpp` | `llama-server` | GGUF (text + vision) | macOS / Linux / Windows |

Each engine takes a **different CLI**, so the command line is built per engine in
`config/flags.py` — passing one engine's flag to another aborts its argparse. `mlx-lm` and
`mlx-vlm` are table-driven (`_tables`); `vllm-mlx` and `llama-cpp` have their own arg builders
(`_vllm_mlx_args`, `_llama_cpp_args`) because their CLIs differ (a `serve` subcommand / short
GGUF flags). **`flags.py` is pure — no I/O.** Filesystem resolution (e.g. a GGUF *folder* →
the `.gguf` *file*, finding a sibling `mmproj`) lives in `server/discovery.py`.

**To add an engine:** extend the `Engine` literal + add fields (`config/models.py`); add an
arg builder + dispatch + `_DEFAULT_BINARY` (`config/flags.py`); map the binary
(`server/discovery.py:SERVER_BINARIES`); add the editor Select option, options block, and
gating (`screens/editor.py`); a setup detect/install entry (`screens/setup.py` +
`bootstrap.py`); and tests (`tests/test_flags.py`, `tests/test_server.py`).

## Package map

- `app.py` — the Textual `App`: theme, global CSS, screen wiring, the running-manager
  registry, clipboard, and clean shutdown of server subprocesses.
- `screens/` — one file per screen: `dashboard`, `editor` (server profiles), `running`,
  `setup`, `chat` (the large one), `mcp_manager`, `skills_manager`, `skill_editor`,
  `project_editor`, `subagent_editor`, `theme_picker`, `xcode_help`.
- `server/` — `manager.py` (one server subprocess: spawn / stream logs / detect readiness /
  stop), `discovery.py` (locate binaries, port checks, GGUF path resolution), `readiness.py`
  (HTTP `/health` + `/v1/models` probe).
- `config/` — `models.py` (`ServerConfig` + settings, pydantic), `flags.py` (the argv
  builder), `store.py` (atomic JSON at `~/.config/mlx-launcher/servers.json`).
- `chat/` — the chat engine: `client.py` (streaming, `<think>`/Harmony parsing, tool-call
  recovery), `acp/bridge.py` (HTTP to the OpenAI API), `tools.py` (web_search), `fs_tools.py`
  (sandboxed file tools), `mcp_client.py` (MCP sessions), `prompted_tools.py` (prompted
  tool-call protocol), `capabilities.py`, `skills.py`, `store.py`, `models.py`.
- `acp/` — the Agent Client Protocol agent for Xcode (`agent.py`, `bridge.py`, `entry.py`).
- `widgets/` — small reusable widgets (toggle chips, code blocks, banner, safe content).
- `skills/` — bundled `SKILL.md` instruction sets shipped inside the package.

## Dev workflow

```sh
./run.sh                        # macOS/Linux: create .venv, install -e ., launch the TUI
./run.sh --reinstall            # force a dependency reinstall
.venv/bin/python -m pytest -q   # run the suite (fast, offline, no real model servers)
```

- Tests are **hermetic and offline** — no network, no real servers. Anything that boots the
  full app (`MlxLauncherApp().run_test()`) MUST first set
  `monkeypatch.setenv("XDG_CONFIG_HOME", tmp_path)`, or it reads/writes the user's real
  `~/.config/mlx-launcher/` (the chat screen opens the most-recent real chat and persists to
  it). There is a real-`llama-server` smoke path used during development, but the committed
  suite never depends on a model.
- The MLX engines run under a **separate interpreter** (installed via `uv tool` / Homebrew),
  not this app's venv — `import mlx_lm` will fail here by design. The app launches engines via
  their console scripts resolved with `shutil.which`, never `python -m`.

## Conventions & gotchas (each cost real debugging time)

- **Textual markup safety.** `Static`/`Label`/`Select` labels, widget `border_title`, and
  `App.notify` parse their string as Textual markup and raise on text like `[w=600&h=400]`
  (everywhere in model output, URLs, JSON, paths). Escaping is unreliable. For ANY
  externally-sourced text, build content from literals via `widgets/safe_content.py`
  (`plain`, `title_sub`) or `Content.assemble(...)`. `App.notify` already defaults
  `markup=False`.
- **One system message.** Fold all system guidance into a single leading system message via
  `client.prepend_system`. Two leading system turns make some chat templates (Qwen) return a
  500.
- **Tool calling is native-first with a prompted-protocol fallback** (`chat/prompted_tools.py`),
  and gpt-oss/Harmony tool calls are recovered from raw text (`chat/client.py`:
  `parse_harmony*`, `recover_*`). These recoveries are deliberately conservative to avoid
  treating prose as a call — don't loosen the regexes without re-running the tests.
- **Per-pane chat concurrency.** The chat runs a main pane + an optional subagent side pane
  with independent generation state (`_gen` / `_cancel_flags` dicts keyed `"main"`/`"side"`).
  Don't reintroduce screen-wide streaming state.
- **Cross-platform processes.** `server/manager.py` and `chat/fs_tools.py` branch on
  `sys.platform == "win32"` (a new process *group* via `CREATE_NEW_PROCESS_GROUP` vs a POSIX
  session; `taskkill` / `proc.terminate()` vs `os.killpg`). **`os.kill(pid, 0)` *terminates*
  the process on Windows** — never use it as a liveness probe there.
- **Each store is one JSON document** written atomically (temp file → flush + fsync → rename).
  A screen that holds its own `store.load()` copy must re-read immediately before saving, or
  it can clobber writes made elsewhere.

## Where common changes go

- **A chat tool:** a spec + runner in `chat/tools.py` (or `fs_tools.py` / `mcp_client.py`),
  wired into the loop in `screens/chat.py:_generate_tools`.
- **A model capability heuristic:** `chat/capabilities.py` (name-based; always user-overridable).
- **Install / run scripts:** `install.sh` / `run.sh` (macOS), `install-linux.sh` /
  `run-linux.sh`, `install-windows.ps1` / `run-windows.ps1`.

## Project conventions

- The app launches and manages **its own** servers — there is intentionally no "connect to an
  external server" feature.
- Prefer **clear text labels** over glyph-only icons in the UI.
- It's **local-first**: no cloud calls or API keys (web_search uses DuckDuckGo without a key).
