# LIS — Local Inference Server

A Claude-Code-styled terminal app (TUI) to launch and manage local model servers — then chat
with them or wire them into Xcode 27. It drives four interchangeable backends:
[**mlx-lm**](https://github.com/ml-explore/mlx-lm) (text) and
[**mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) (vision) on Apple Silicon,
**vllm-mlx** (vLLM-style MLX), and [**llama.cpp**](https://github.com/ggml-org/llama.cpp)
(`llama-server`, GGUF models) on **macOS, Linux, or Windows**.

<p align="center">
  <img src="preview.png" alt="LIS chat UI — projects/chats sidebar, skill and server pickers, context bar, and plan/reason/web/tools toggles" width="820">
</p>

- **Pick the engine** per profile: **mlx-lm** (text), **mlx-vlm** (vision-language),
  **vllm-mlx** (vLLM-style: continuous batching, KV-quant, native tools), or **llama.cpp**
  (GGUF; runs on any platform). The launcher builds each engine's correct command line and
  gates the UI to the flags that engine actually accepts.
- **Drag & drop** a model folder onto the terminal (or paste a HuggingFace repo id). For
  **llama.cpp**, point at a `.gguf` file, its model folder (the right `.gguf` is picked
  automatically), or an HF repo — and a sibling vision projector (`mmproj`) loads on its own.
- Tweak options (temperature, max-tokens, top-p/k, prompt cache, …) plus a free-form
  **custom params** box for anything else (e.g. quantized-KV-cache flags on servers that
  support them).
- **Launch** the server, see its address (`http://host:port/v1`) and **live logs**.
- **Save** named server profiles and quick-launch them.
- Connect to **Xcode 27** two ways: an OpenAI-compatible *Locally Hosted* provider, and
  an **ACP** (Agent Client Protocol) stdio agent (with agentic file edits).
- **Chat** with a running server in a built-in Claude-style UI (press `c`): projects +
  chats sidebar (create/delete with confirmation), streaming replies that render
  **Markdown with syntax-highlighted, copyable code / JSON blocks**, a thinking panel for
  reasoning models (with a per-chat **reasoning-effort** control), a tok/s footer,
  **drag-and-drop file attachments** (images for vision
  models, text for any), a multiline prompt (Enter sends · Shift+Enter newline),
  regenerate / edit-last / export-to-Markdown, and a live **theme picker** (Ctrl+T).
- **Tool use in chat** (toggle *tools* in the chat): the model can call a built-in
  **web_search** (DuckDuckGo, no API key) and tools from any **MCP servers** you connect
  (stdio or SSE) — manage them with `m` on the dashboard or Ctrl+G in chat. Tool calling is
  native-first with a prompted-protocol fallback, so it works across models (Qwen, Gemma,
  Nemotron, GPT-OSS, MiniMax, Step, …).
- **Subagents**: define named specialist agents (own model + system prompt + web/MCP/skills +
  an uploaded **knowledge base** of docs/PDFs that's always in their context) and open one as a
  50/50 **side chat** beside the main model — message either pane.
- **Code in a folder**: set a project's **working directory** (`+ Project` / Ctrl+E) and the
  model gets file tools — `read` / `write` / `edit` / `delete` / `run_command` — scoped to
  that folder (paths can't escape it), with an **approve / deny prompt** before anything
  mutating. It reads `AGENTS.md` first if present.
- **Skills**: pick a `SKILL.md` instruction set per chat — bundled platform skills, your own
  **custom** ones, or installed **BMAD** skills; browse/create/install with `k`.
- **Plan mode**: a per-chat toggle that makes the model produce a plan for you to approve
  instead of taking action.
- **Context bar**: a live token-usage meter showing how much of the model's context window the
  conversation uses.
- **Talk to it** (optional): a **🎙 Mic** button transcribes your speech into the prompt
  (Whisper — `mlx-whisper` on Apple Silicon, `faster-whisper` elsewhere) and a **🔊 Read aloud**
  button speaks the last reply (Kokoro-82M via `kokoro-onnx`, with the system voice as a
  fallback). All local, no API keys. Enable with `pip install "mlx-launcher[voice]"` (or the
  **Install voice** button on the setup screen). Auto-send-after-mic and auto-read-replies are
  opt-in settings.
- **Dependency self-check**: detects which engine binaries are on your `PATH`
  (`mlx_lm.server`, `mlx_vlm.server`, `vllm-mlx`, `llama-server`) and offers to install the
  ones you're missing (`p` on the dashboard).
- **Global install**: run it from anywhere like `claude`.

## Quick start

```sh
./run.sh                 # macOS / Linux
.\run-windows.ps1        # Windows (PowerShell)
```

First run creates a `.venv` and installs the app's pure-Python deps (Textual, httpx,
agent-client-protocol, …). It does **not** install the model runtimes — the app detects which
engine binaries are on your `PATH` and offers to install the ones you're missing.

## Install globally

```sh
./install.sh                                                     # macOS
./install-linux.sh                                              # Linux
powershell -ExecutionPolicy Bypass -File .\install-windows.ps1   # Windows
lis-start                                                            # then launch from anywhere
```

Each script installs the launcher (pipx if available, else a local `.venv` + `~/.local/bin`
symlinks) and exposes the `lis-start` command. The **Linux** and **Windows** scripts also fetch a
prebuilt **`llama-server`** from the llama.cpp releases — MLX is Apple-Silicon-only, so
llama.cpp is the engine on those platforms.

## Uninstall

```sh
./uninstall.sh                                                   # macOS / Linux
powershell -ExecutionPolicy Bypass -File .\uninstall-windows.ps1  # Windows
```

Removes the global command — both the old `mlxs` / `mlx-launcher` and the new `lis-start` —
along with the pipx install and the local `.venv`. Your server profiles and chats
(`~/.config/mlx-launcher/`) are kept; add `--purge` (`-Purge` on Windows) to delete those too.
The model engines (MLX / llama.cpp) are left installed.

## Requirements

- **Apple Silicon (macOS)** for the MLX engines (mlx-lm, mlx-vlm, vllm-mlx); **macOS, Linux,
  or Windows** for the llama.cpp engine. Python 3.10–3.14.
- A model runtime for whichever engine you use:
  - **mlx-lm / mlx-vlm / vllm-mlx** — installed via `uv tool` or the in-app setup (`p`).
  - **llama.cpp** (`llama-server`) — `brew install llama.cpp` on macOS; the Linux/Windows
    install scripts fetch a prebuilt binary for you.
- **Voice (optional)** — for the chat mic / read-aloud buttons: `pip install "mlx-launcher[voice]"`
  (`sounddevice` + `numpy` + `kokoro-onnx`, plus `mlx-whisper` on Apple Silicon / `faster-whisper`
  elsewhere). The Kokoro voice model (~325 MB) downloads once on first use. Read-aloud also works
  without the extra via the OS voice (macOS `say` / `espeak-ng`).
- [**Homebrew**](https://brew.sh) is recommended on macOS — the easiest way to get Python, and
  `./install.sh` uses it for a clean global install.
