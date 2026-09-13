# HTTP API

Every route is behind the bearer token and the rate limiter, except the
three marked open below. The web UI is a client of this API and has no
privileged path of its own.

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
| `GET` | `/memory` | optional | List the vector store **as stored**, no query and no embedding call — `search` is semantic by construction, so it cannot tell you what is in there without a question (v3.14). `?kind=` filters, and the breakdown by kind comes with it |
| `DELETE` | `/memory/{id}` | optional | Forget one entry, from both tables. The deliberate step `deploy/rag_resplit.py` stops short of: it reports the entries that are a refusal and nothing else, and deletes none of them |
| `GET` | `/history` | optional | Full rolling history with stable ids (v3.9) |
| `GET` | `/drawer` | optional | Currently pinned messages, the "tiroir" (v3.9) |
| `POST` | `/drawer/pin` | optional | Pin a message by id — pins its exchange partner too (v3.9) |
| `POST` | `/drawer/unpin` | optional | Unpin a message by id, independently of its partner (v3.9) |
| `POST` | `/compact` | optional | Force a context compaction pass now (v3.9) |
| `GET` | `/context` | optional | What the next prompt will weigh — the gauge behind the header readout (v3.12) |
| `GET` | `/jobs` | optional | Every delegation job and its state (v3.13). Deliberately not the day-to-day way to read one: the conversation thread is, per the zero-tab rule. This exists so a job can be inspected without reading `data/jobs.json` over SSH |
| `POST` | `/pair/claim` | open | Exchange a single-use `!pair` token for the bearer token. Open because it is where a device **gets** its credential — requiring one would make pairing impossible |
| `GET` | `/docs` | open | Interactive API docs (Swagger) |

**Auth:** set `API_TOKEN` in the environment to require
`Authorization: Bearer <token>` on every "optional" route above. Unset (the
default), the API is exactly as open as before this existed — nothing changes
unless you opt in. `/`, `/health` and `/pair/claim` always stay open — the UI
shell, monitoring probes, and the one endpoint a device with no credential yet
has to be able to call. What guards `/pair/claim` is the token it is handed
rather than the one it does not ask for: 256 bits, single use, five minutes,
and the rate limiter, which is what makes guessing it pointless rather than
merely improbable. Unknown, already-claimed and expired tokens all answer with
the same `401` and the same sentence, because naming which one it was would
confirm to a prober that a token existed. The web UI has a 🔑 **Token** button in the header that
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

### `POST /chat` response

| Field | Type | Note |
|---|---|---|
| `output` | `str` | The answer |
| `tool` | `str` | Capability that produced it |
| `ok` | `bool` | A rendering directive, not a verdict — see `not_answered` in the trace |
| `steps` | `int` | Steps taken |
| `error` | `str \| null` | |
| `usage` | `object \| null` | Absent when no accounting scope was open |
| `job_id` | `int \| null` | The delegation job this turn created, advanced, launched or cancelled |

`job_id` is **absent rather than zeroed** when the turn was about no job —
the same convention as `usage`, so a client can tell "no job" from "job
number 0". It is set by every delegation turn that names one job: the
`delegate` graph creating it, and the interception in `delegation.py`
answering its questions, approving, launching or cancelling it. The one
delegation turn that reports nothing is `jobs`, the listing, which is
about all of them.

It exists because a delegation is long-running work the caller has to
follow after the response: poll `GET /jobs` for this id and notify when
it finishes. Before it, the id was only inside the answer's prose (`"Job
12 lancé."`), so a client recovered it with a regex over French text that
would break the day any of those sentences is reworded.

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
