"""Read a subagent's 'knowledge base' (uploaded docs) into a text block injected into
its system prompt — so the content is in the model's context whenever the subagent is
used. This is full-context injection (not embeddings): simple, local — but capped so a
large doc set can't blow the context window.

Single files added explicitly are read as text; **PDFs are text-extracted via pypdf**.
Folders are walked for text/markdown/code files and PDFs (other binaries/images are
skipped — they'd be garbage decoded as text)."""

from __future__ import annotations

import os

MAX_FILE_BYTES = 200_000      # per file
MAX_TOTAL_BYTES = 600_000     # whole knowledge base (keeps the injected prompt bounded)
MAX_FILES = 50

# Extensions read when walking a folder.
_TEXT_EXT = {
    ".md", ".markdown", ".txt", ".text", ".rst", ".json", ".jsonl", ".csv", ".tsv",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".html", ".htm", ".xml", ".log",
    ".tex", ".org", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".zsh", ".sql",
    ".swift", ".kt", ".scala", ".lua", ".pl", ".r",
}
# PDFs are read via text extraction (a special case in _read), not raw decode.
_PDF_EXT = {".pdf"}
# Extensions read when walking a folder = text files + PDFs.
_FOLDER_READABLE = _TEXT_EXT | _PDF_EXT
# Never read these, even when added explicitly (they'd be garbage decoded as text).
_BINARY_EXT = {
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".zip", ".tar", ".gz",
    ".7z", ".rar", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".mov", ".wav", ".bin", ".exe", ".dll", ".so", ".dylib", ".gguf",
    ".safetensors", ".npz", ".pkl",
}


def _read_pdf(path: str, limit: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""  # pypdf missing → skip the PDF rather than inject nothing/garbage
    try:
        parts, total = [], 0
        for page in PdfReader(path).pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
                total += len(t)
                if total >= limit:
                    break
        text = "\n".join(parts).strip()
    except Exception:  # noqa: BLE001 — encrypted/corrupt/scanned PDF, etc.
        return ""
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n… (truncated)"
    return text


def _read(path: str, limit: int) -> str:
    if os.path.splitext(path)[1].lower() in _PDF_EXT:
        return _read_pdf(path, limit)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(limit + 1)
    except OSError:
        return ""
    if len(data) > limit:
        data = data[:limit].rstrip() + "\n… (truncated)"
    return data


def _iter_files(paths: list[str]):
    """Yield (display_name, full_path) for each readable file in `paths` (files or folders)."""
    for p in paths:
        path = os.path.expanduser(p or "")
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() not in _BINARY_EXT:
                yield os.path.basename(path), path
        elif os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for fn in sorted(files):
                    if os.path.splitext(fn)[1].lower() in _FOLDER_READABLE:
                        full = os.path.join(root, fn)
                        yield os.path.relpath(full, path), full


def load_knowledge(paths: list[str]) -> str:
    """Concatenate the knowledge files into one `<doc>`-framed block (capped), or '' if
    there's nothing readable."""
    if not paths:
        return ""
    docs: list[str] = []
    total = 0
    for name, full in _iter_files(paths):
        if len(docs) >= MAX_FILES or total >= MAX_TOTAL_BYTES:
            break
        body = _read(full, min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total))
        if not body.strip():
            continue
        docs.append(f'<doc name="{name}">\n{body}\n</doc>')
        total += len(body)
    if not docs:
        return ""
    return (
        "# Knowledge base\n\n"
        "Reference material provided to you below. Treat it as authoritative for this "
        "conversation, prefer it over your own assumptions, and cite the document name "
        "when you rely on it.\n\n" + "\n\n".join(docs)
    )
