"""Enforce the layered architecture: the dependency arrows point only inward.

    models  ←  engine  ←  core        (client is self-contained; reaches the backend over the wire)

These tests parse every module's imports (absolute + relative resolved) and fail the build if a
layer reaches "upward" or sideways into a layer it must not know about. This is what makes the
format engine swappable and the backend frontend-agnostic — the boundary is mechanically checked,
not just a convention.

(The frontends — screens/, app.py, acp/ — still reach the backend in-process for a few paths that
haven't moved over the wire yet, e.g. the subagent side-pane and context compaction; those are
tracked separately and intentionally not asserted here.)"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "mlx_launcher"
_LAYERS = {"engine", "core", "client", "models", "screens", "acp", "chat", "config", "server"}


def _imported_subpackages(pyfile: pathlib.Path) -> set[str]:
    """The set of ``mlx_launcher.<subpkg>`` names this file imports, resolving relative imports
    against the file's own package."""
    rel = pyfile.relative_to(SRC).with_suffix("")
    pkg_parts = ["mlx_launcher", *rel.parts[:-1]]
    out: set[str] = set()
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mlx_launcher."):
                    out.add(alias.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("mlx_launcher."):
                    out.add(node.module.split(".")[1])
            else:  # relative: resolve against this file's package
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                full = [*base, *(node.module.split(".") if node.module else [])]
                if len(full) >= 2 and full[0] == "mlx_launcher":
                    out.add(full[1])
    return {s for s in out if s in _LAYERS}


def _layer_files(layer: str):
    return sorted((SRC / layer).rglob("*.py"))


def test_models_is_the_leaf():
    forbidden = _LAYERS - {"models"}
    for f in _layer_files("models"):
        bad = _imported_subpackages(f) & forbidden
        assert not bad, f"{f.relative_to(SRC)} imports {bad}; models/ must depend on nothing"


def test_engine_depends_only_on_models():
    forbidden = {"core", "client", "screens", "acp", "chat", "config", "server"}
    for f in _layer_files("engine"):
        bad = _imported_subpackages(f) & forbidden
        assert not bad, f"{f.relative_to(SRC)} imports {bad}; engine/ may depend only on models/"


def test_core_depends_only_on_engine_and_models():
    forbidden = {"client", "screens", "acp", "chat", "config", "server"}
    for f in _layer_files("core"):
        bad = _imported_subpackages(f) & forbidden
        assert not bad, (f"{f.relative_to(SRC)} imports {bad}; core/ may depend only on engine/ + "
                         "models/ (use core.persistence / core.server, not the chat/config/server shims)")


def test_client_reaches_backend_over_the_wire_not_by_import():
    for f in _layer_files("client"):
        bad = _imported_subpackages(f) & {"engine", "core"}
        assert not bad, f"{f.relative_to(SRC)} imports {bad}; client/ must use the HTTP+SSE API only"
