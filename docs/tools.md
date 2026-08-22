# Tools

What each tool does, what it refuses to do, and what it costs. A tool
needs both a `run()` handler and an `ENABLED_TOOLS` opt-in -- code
existing is never enough to make it reachable.

## Tools

| Tool | Activated by | Description |
|---|---|---|
| `chat` | default | Conversational response |
| `code` | default | Code generation |
| `files` | `ENABLED_TOOLS=chat,code,files` | Sandboxed read/write/list within `WORKSPACE_DIR` |
| `shell` | `ENABLED_TOOLS=chat,code,shell` | Subprocess execution within `WORKSPACE_DIR` + `SHELL_ALLOWED_COMMANDS` |
| `git` | `ENABLED_TOOLS=chat,code,git` | Read-only git operations (status, diff, log, show, branch) — deliberately never gains a write counterpart reachable by the router: a commit/push has a real cost if the router hallucinates, so any git write stays a separate, human-confirmed flow outside tool dispatch, not a router decision |
| `memory` | `ENABLED_TOOLS=chat,code,memory` | Router-dispatchable RAG remember/recall (v3.7) |
| `test` | `ENABLED_TOOLS=chat,code,test` | Sandboxed pytest/ruff runner, own allowlist (`TEST_ALLOWED_COMMANDS`) separate from the shell tool's |
| `review` | `ENABLED_TOOLS=chat,code,review` | Reads a file (optionally runs its tests first) and returns an LLM analysis — "relis X et donne ton avis", not just "lis X" (see the Router reachability note below on that exact ambiguity) |
| `web_fetch` | `ENABLED_TOOLS=chat,code,web_fetch` | Fetches a URL you already know — no search capability, SSRF-guarded, best-effort HTML→text extraction |
| `web_search` | `ENABLED_TOOLS=chat,code,web_search` | Ranked links/snippets from a self-hosted SearXNG instance — no synthesis, just the list |
| `research` | `ENABLED_TOOLS=chat,code,research` | Search → fetch top results → synthesize one answer, as a single deterministic call (see below) |
| `sysadmin` | `ENABLED_TOOLS=chat,code,sysadmin` | Discover → collect → synthesize: diagnoses a service/system problem from real logs, read-only always — never restarts/stops anything. Works kernel-log-only out of the box; see [`deploy/README.md`](../deploy/README.md) for read-only proxies giving it real `journalctl`/`systemctl`/`podman` access |
| `recall` | `ENABLED_TOOLS=chat,code,memory,recall` | Search vector memory → synthesize an answer, as one deterministic call. Same reasoning as `research` versus `web_search`: `memory`'s own recall returns entries, this returns a sentence. Calls `memory.search()` directly, so `memory` need not also be in `ENABLED_TOOLS` — but its embedding server does need to be reachable |
| `delegate` | `ENABLED_TOOLS=chat,code,delegate` | Opens a delegation to Claude Code (v3.13): drafts a spec, asks follow-up questions when detail is missing, and hands off once you approve. Only the *entry* point is a tool — answering the questions, approving and cancelling happen above the router in `delegation.py`, so adding `delegate` here does not change how those are handled |

A tool is only dispatchable if it has a `run()` function **and** appears in `ENABLED_TOOLS`.
Implementing `run()` in a module is not enough — the opt-in is intentional for tools with side effects.

**Router reachability (v3.5):** the router's own prompt and validation are generated from
`ENABLED_TOOLS` — every enabled tool is offered as a routing option in normal conversation,
not only via an explicit [Graph](architecture.md) (`POST /run`). Before v3.5, `files`/`shell`/`git`
were reachable only through a Graph even when enabled, because the router's prompt and JSON
validation hardcoded exactly `{"chat", "code"}` regardless of `ENABLED_TOOLS`. Nothing about the
opt-in itself changed: a tool still has to be listed in `ENABLED_TOOLS` to be reachable either way,
and each tool's own sandboxing (allowlist, timeout, `WORKSPACE_DIR` confinement, git's read-only
subcommand list) applies the same regardless of how it's invoked.

**Grammar-constrained decoding (v3.6, llama.cpp only):** by default, the router's llama.cpp
requests include a GBNF grammar ([`router/grammar.py`](../src/forge/router/grammar.py)) that
constrains sampling to the router's exact JSON schema — the model cannot emit tokens for
hallucinated dialogue turns, leaked prompt text, or malformed JSON, because those tokens simply
aren't valid at that point in the grammar. This was added after a real failure mode: with a long
enough prompt (many enabled tools) and a stale conversation history in context, a model would
occasionally answer a *fictional* follow-up question instead of the real one, or emit nothing
usable at all — the router's fallback chain caught it, but a placeholder isn't a good answer.
Grammar constraint stops that class of failure before it starts, at the cost of being provider-
specific (only llama.cpp exposes raw GBNF sampling this way — Ollama has a coarser `"format":
"json"`, OpenRouter has `response_format`, neither can pin `tool` to a specific set of literal
values). Set `LLAMA_CPP_USE_GRAMMAR=false` if your server version doesn't support the `grammar`
completion field, or to rule it out while debugging — the prompt-engineering + parser fallback
chain underneath it all is unchanged and still does the same job on its own, just with a higher
failure rate on a stressed prompt.

**Why `research` exists alongside `web_search` (v3.10):** a plain search only returns links and
snippets — turning that into an actual synthesized answer needs a second step (fetch a promising
result, then have the model write a real answer from it). Asking the router to decide that second
step itself proved unreliable in practice with a small local model: even with an explicit worked
JSON example showing exactly what to do next, it would sometimes just repeat the identical search
call instead, tripping the loop guard. Disabling `LLAMA_CPP_CACHE_PROMPT` and reproducing the same
failure ruled out a KV-cache bug — this is a genuine limit at multi-step self-correction for this
model class, not a fixable prompt or infra issue. `research` (`graphs/research.py`) removes the
decision from the router's hands entirely: search → fetch the top `RESEARCH_FETCH_TOP_N` results →
one synthesis call, run as a fixed sequence inside a single dispatchable call, the same pattern
already used by the `review` graph. `web_search` stays for when the user genuinely wants a list of
links/sources rather than an answer.

**Why `/chat` isn't streamed (yet):** for `tool="chat"`, the router's single LLM call already
*is* the answer — `content` in `{"tool":"chat","content":"..."}` is generated in the same call as
the routing decision, and `tools/chat.py` just returns it unchanged. Streaming that content would
mean streaming tokens before the JSON (and therefore the tool choice) is even complete — and the
parser deliberately prefers the *last* complete JSON object it finds, not the first, because small
local models sometimes echo earlier conversation before producing the real answer. Streaming
token-by-token would risk showing stale/wrong content that then gets silently replaced — worse
UX than no streaming. Real streaming needs decision and generation split into two LLM calls (a
fast classify-only call, then a separate streamed generation call once the tool is known) — a
real latency trade-off on already-slow local hardware. Deferred, no version scheduled yet:
v3.7 went to vector memory/RAG instead (see below), on the reasoning that Forge not knowing
the user's own decisions/projects mattered more than response latency.

---

[← Documentation index](README.md) · [← Project README](../README.md)
