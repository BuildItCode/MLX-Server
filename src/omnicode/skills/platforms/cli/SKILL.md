---
name: cli
description: >
  CLI tool code quality rules. Loaded automatically when platform=cli.
  Covers command structure, UX, error output, and shell compatibility.
---

# CLI Quality


## Command Structure

- Commands are nouns or verb-noun pairs: `user create`, `config set` — not `createUser`
- Flags follow POSIX conventions: short `-f`, long `--flag`, boolean flags have `--no-{flag}` negation
- `--help` and `--version` always available on root and every subcommand
- `--dry-run` on any command that mutates state
- Required positional args before optional ones; no more than 2 positional args — use flags beyond that

## Output

- Stdout: machine-readable output (results, data)
- Stderr: human-readable status, progress, errors — never mix
- `--json` flag on all commands that return data — outputs clean JSON to stdout
- `--quiet` / `-q` suppresses all non-error output
- Colours via a library (`chalk`, `lipgloss`, `click.style`) with auto-detection of TTY; disabled with `NO_COLOR=1`
- Progress indicators only when operation > 2s; spinner on stderr

## Error Handling

- Exit code `0` = success, `1` = user error, `2` = system/unexpected error — always consistent
- Error messages start with the problem, then the fix: "Config file not found. Run `{cmd} init` to create one."
- Never print a stack trace to end users — log to a debug file or behind `--debug`

## Configuration

- Config precedence: CLI flags > env vars > config file > defaults — documented and consistent
- Config file location follows XDG Base Directory spec (`~/.config/{tool}/`)
- Sensitive values never stored in config files — document which vars use env

## Validation Checks

```
✓ --help available on all commands
✓ Stdout/stderr separation
✓ Consistent exit codes
✓ --json flag on data-returning commands
✓ No stack traces in user-facing errors
```
