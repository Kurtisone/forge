# Forge

[![CI](https://github.com/Kurtisone/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Kurtisone/forge/actions/workflows/ci.yml)

Forge is a lightweight LLM-based agent runtime built around a router + tool execution model.
Instead of relying on a monolithic prompt or complex reasoning loops, Forge delegates actions
to explicit tools selected by a structured LLM router.

It runs on the machines it is written on — a Steam Deck and a small home server —
which is what most of the design decisions here are downstream of.

---

### Core Concept

```
User Input
   ↓
LLM Router  (structured JSON decision)
   ↓
Capability Registry  →  Policy Engine
   ├── chat        (conversational response)
   ├── code        (code generation)
   ├── files       (sandboxed read/write/list)
   ├── shell       (sandboxed subprocess)
   ├── git         (read-only git operations)
   ├── memory      (remember/recall, vector search)
   ├── recall      (search memory → synthesize, one call)
   ├── test        (sandboxed pytest/ruff runner)
   ├── review      (read a file, optionally test it, analyze)
   ├── web_fetch   (fetch a known URL)
   ├── web_search  (SearXNG links/snippets, no synthesis)
   ├── research    (search → fetch → synthesize, one call)
   ├── sysadmin    (discover → collect → synthesize, read-only diagnosis)
   └── delegate    (draft a spec, hand off to Claude Code)
```

The model must output a strict JSON instruction (`{"tool": "...", "content": "..."}`)
describing which tool to invoke. The router is resilient: it handles JSON,
XML tool-call format (Qwen HERETIC), markdown code fences, and plain text as
fallbacks, in that order. Repeated tokens, leaked prompt instructions, and
empty outputs are detected and replaced with a clean placeholder.

---

### Quick start

```bash
git clone https://github.com/Kurtisone/forge.git
cd forge
cp .env.example .env          # set API_TOKEN, at minimum
pip install -r requirements.txt

PYTHONPATH=src python -m forge.main          # REPL
PYTHONPATH=src uvicorn forge.api:app         # HTTP API + web UI on :8000
```

Or with a container:

```bash
podman compose build forge
podman compose up -d --no-deps forge
```

Forge refuses to start without `API_TOKEN` — that is deliberate, not a
papercut. Full instructions in [docs/usage.md](docs/usage.md).

---

### Documentation

Everything past "how do I start it" lives in [docs/](docs/README.md):

- [Architecture](docs/architecture.md) — how a turn flows, and why the boundaries are where they are
- [Usage](docs/usage.md) — REPL, web UI, CLI, container
- [Configuration](docs/configuration.md) — the settings you are likely to touch
- [Tools](docs/tools.md) — what each tool does, refuses, and costs
- [Memory, RAG and traces](docs/memory.md) — the three stores and how they differ
- [HTTP API](docs/api.md) — routes, auth, delegation jobs
- [Development](docs/development.md) — tests, CI, measurement harnesses
- [Version history and roadmap](docs/roadmap.md) — what landed when

Plus, at the root: [ARCHITECTURE.md](ARCHITECTURE.md) for the long-term
direction, [SECURITY.md](SECURITY.md) for the threat model, and
[.env.example](.env.example) as the exhaustive configuration reference.

---

### Security

Forge dispatches model-chosen tools on your own machine, sometimes
against data it fetched from elsewhere. [SECURITY.md](SECURITY.md)
states the threat model it is built for (one operator, one machine,
private network, public repo), what is enforced deterministically in
code rather than asked of the model, and the limits that are known and
accepted -- including the one worth reading before you edit
`ENABLED_TOOLS`: `files` and `test` together are equivalent to
`shell`.

---

### Status

Forge is an experimental local runtime, not a production framework.
The public API (orchestrator, tool registry, providers, graph engine) is stabilising from v3.0 onward.

---

### License

MIT
