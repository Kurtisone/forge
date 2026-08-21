# HTTP API

Every route is behind the bearer token and the rate limiter. The web UI
is a client of this API and has no privileged path of its own.

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | open | Web UI |
| `GET` | `/health` | open | Provider + model info (for `llama_cpp`, the actually-loaded model, queried live from llama-server — see below) |
| `POST` | `/chat` | optional | Single conversation turn |
| `POST` | `/review` | optional | File content analysis, optionally running its tests first (`test_path` field, v3.10) |
| `POST` | `/run` | optional | Run any graph by name |
| `GET` | `/tools` | optional | Tools that are enabled **and** permitted right now, what the active policy is subtracting (`denied`), and available graphs — the same answer the router is given |
| `GET` | `/traces?n=10` | optional | Recent execution traces |
| `POST` | `/remember` | optional | Store a decision/todo in vector memory (v3.7) |
| `GET` | `/search?q=...` | optional | Semantic search over remembered decisions/todos |
| `GET` | `/history` | optional | Full rolling history with stable ids (v3.9) |
| `GET` | `/drawer` | optional | Currently pinned messages, the "tiroir" (v3.9) |
| `POST` | `/drawer/pin` | optional | Pin a message by id — pins its exchange partner too (v3.9) |
| `POST` | `/drawer/unpin` | optional | Unpin a message by id, independently of its partner (v3.9) |
| `POST` | `/compact` | optional | Force a context compaction pass now (v3.9) |
| `GET` | `/context` | optional | What the next prompt will weigh — the gauge behind the header readout (v3.12) |
| `GET` | `/jobs` | optional | Every delegation job and its state (v3.13). Deliberately not the day-to-day way to read one: the conversation thread is, per the zero-tab rule. This exists so a job can be inspected without reading `data/jobs.json` over SSH |
| `GET` | `/docs` | open | Interactive API docs (Swagger) |

**Auth:** set `API_TOKEN` in the environment to require
`Authorization: Bearer <token>` on every "optional" route above. Unset (the
default), the API is exactly as open as before this existed — nothing changes
unless you opt in. `/` and `/health` always stay open, for the UI shell and
monitoring probes. The web UI has a 🔑 **Token** button in the header that
prompts for the token and remembers it (localStorage) for subsequent requests.

**Rate limiting:** the same "optional" routes are also behind an in-memory
sliding-window limiter — `RATE_LIMIT_REQUESTS` per `RATE_LIMIT_WINDOW_SECONDS`
per client IP (default: 30 per 60s), `429 Too Many Requests` with a
`Retry-After` header past that. No external service (no redis) — a plain
process-local counter, single-worker only: running uvicorn with multiple
workers gives each its own counter. Set `RATE_LIMIT_ENABLED=false` to disable,
e.g. behind a proxy that already rate-limits.

**`POST /run` example:**
```json
{ "graph": "review", "input": "src/forge/main.py", "context": {"question": "Security issues?"} }
```

## Delegation Jobs

`GET /jobs` lists every delegation job and its state. Like every other
endpoint it requires the bearer token, so typing the URL into a browser
returns 401 -- the address bar cannot send a header. Use curl:

```bash
curl -s -H "Authorization: Bearer $FORGE_TOKEN" localhost:8000/jobs
```

This is a debugging view, not the interface. A job is meant to be read and
answered in the conversation itself; if you find yourself reaching for this
endpoint to find out what a job is waiting on, that is the thread failing to
say so, and the fix belongs there.

---

[← Documentation index](README.md) · [← Project README](../README.md)
