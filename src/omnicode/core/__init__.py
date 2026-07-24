"""The **backend** layer: agent orchestration, sessions, tools, token budgeting +
compaction, persistence, model-server supervision, and (in a later step) the local
HTTP+SSE service.

Depends only on the ``engine`` (format engine) and ``models`` layers — never on any
frontend. The TUI and ACP frontends reach this layer over the wire, not by import."""
