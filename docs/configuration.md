# Configuration

Every setting lives in the environment. [`.env.example`](../.env.example)
is the exhaustive reference and carries the reasoning behind each value;
this page covers the ones you are likely to touch.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `FORGE_PROVIDER` | LLM backend: `llama_cpp`, `ollama`, `openrouter` | `llama_cpp` |
| `LLM_MODEL` | Model name. For `ollama`/`openrouter` this is sent with every request and must match a real model. For `llama_cpp` it's **never sent** — llama-server always serves whatever GGUF it was launched with — so this value is only a fallback label for `/health`; `/health` queries llama-server's own `/props` for the live model name first and only falls back to this if that probe fails | `default` |
| `OLLAMA_URL` | Ollama endpoint | `http://127.0.0.1:11434/api/generate` |
| `LLAMA_CPP_URL` | llama.cpp endpoint | `http://127.0.0.1:8080` |
| `LLAMA_CPP_N_PREDICT` | Max tokens per llama.cpp response | `512` |
| `LLAMA_CPP_TIMEOUT` | HTTP timeout for llama.cpp requests (seconds) | `120` |
| `LLAMA_CPP_USE_GRAMMAR` | GBNF grammar-constrained decoding for llama.cpp — forces output to match the router's JSON schema at the sampling level | `true` |
| `LLAMA_CPP_ID_SLOT` | llama-server slot to pin every request to, so its KV cache can be reused across turns (v3.8) | `0` |
| `LLAMA_CPP_CACHE_PROMPT` | Ask llama-server to reuse its KV cache from the previous call's matching prefix (v3.8) | `true` |
| `OPENROUTER_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| `OPENROUTER_API_KEY` | OpenRouter API key | *(empty)* |
| `MAX_STEPS` | Hard ceiling on router→tool steps per run (multi-step only happens if the router sends `"done": false`) | `1` |
| `ENABLED_TOOLS` | Comma-separated allowlist of dispatchable tools | `chat,code` |
| `WORKSPACE_DIR` | Root directory for files + shell tools | `data/workspace` |
| `SHELL_TIMEOUT` | Max seconds for a shell tool command | `30` |
| `SHELL_ALLOWED_COMMANDS` | Comma-separated command allowlist for the shell tool | `ls,cat,head,tail,wc,grep,find,python3,pip,pytest` |
| `MEMORY_ENABLED` | Persist and recall conversation history | `true` |
| `MEMORY_FILE` | Path to the JSON memory file | `data/memory.json` |
| `MEMORY_MAX_HISTORY` | Hard-cap safety net on message count, behind compaction (v3.9) — pinned messages are exempt | `100` |
| `COMPACTION_ENABLED` | Replace old non-pinned messages with a summary once `COMPACTION_THRESHOLD` is crossed, instead of just dropping them (v3.9) | `true` |
| `COMPACTION_THRESHOLD` | Message count that triggers a compaction pass | `80` |
| `COMPACTION_KEEP_RECENT` | Most recent non-pinned messages always left untouched by compaction | `20` |
| `COMPACTION_STRATEGY` | `rag_pointer` (no LLM call, pushes the block into vector memory and leaves a pointer) or `llm_summary` (one LLM call, condenses inline) | `rag_pointer` |
| `TRACE_ENABLED` | Write JSONL execution trace per run | `true` |
| `TRACE_FILE` | Path to the JSONL trace file | `data/traces.jsonl` |
| `SHOW_DEBUG` | Emit full structured trace to stderr (prompt, raw output, timings) | `false` |
| `API_TOKEN` | Bearer token required on `/chat`, `/review`, `/run`, `/tools`, `/traces`. Empty = API stays open | *(empty)* |
| `RATE_LIMIT_ENABLED` | In-memory sliding-window rate limit on the same routes as `API_TOKEN` | `true` |
| `RATE_LIMIT_REQUESTS` | Max requests per client IP per window | `30` |
| `RATE_LIMIT_WINDOW_SECONDS` | Window size in seconds | `60` |
| `EMBEDDING_URL` | Embedding-only llama.cpp endpoint (separate instance from `LLAMA_CPP_URL`) for `/remember`, `/search` (v3.7) | `http://127.0.0.1:8082/embedding` |
| `EMBEDDING_DIM` | Embedding vector dimension, must match the served model | `1024` |
| `EMBEDDING_TIMEOUT` | HTTP timeout for embedding requests (seconds) | `30` |
| `RAG_DB_FILE` | Path to the SQLite-vec vector memory file | `data/forge_rag.db` |
| `TEST_TIMEOUT` | Max seconds for a test/lint tool command | `60` |
| `TEST_ALLOWED_COMMANDS` | Comma-separated command allowlist for the test tool — separate from `SHELL_ALLOWED_COMMANDS` on purpose | `pytest,ruff` |
| `WEB_FETCH_TIMEOUT` | HTTP timeout for `web_fetch` requests (seconds) | `15` |
| `WEB_FETCH_MAX_BYTES` | Raw response byte cap before truncation | `2097152` (2 MiB) |
| `WEB_FETCH_ALLOWED_DOMAINS` | Optional domain allowlist — empty means any public domain, subject to the (non-configurable) SSRF guard | *(empty)* |
| `SEARXNG_URL` | Self-hosted SearXNG instance for `web_search`/`research` — not a cloud API | `http://127.0.0.1:8888` |
| `SEARXNG_TIMEOUT` | HTTP timeout for SearXNG requests (seconds) | `10` |
| `SEARXNG_MAX_RESULTS` | Max results returned per search | `5` |
| `RESEARCH_FETCH_TOP_N` | How many top search results `research` fetches in full before synthesizing | `3` |
| `RESEARCH_FETCH_CHARS_PER_RESULT` | Per-result fetched-content cap fed into the synthesis prompt | `1500` |
| `SYSADMIN_DISCOVERY_TIMEOUT` | Timeout for the discovery step (seconds) | `10` |
| `SYSADMIN_COLLECT_TIMEOUT` | Timeout for each log-collection command (seconds) | `15` |
| `SYSADMIN_LOG_CHARS_BUDGET` | Hard cap on the log block inserted into the synthesis prompt, independent of line count — truncates keeping the **end** of the log, since that is where the recent events are | `2000` |
| `RECALL_MAX_ANSWER_CHARS` | Cap on a `recall` answer. Far smaller than `research`'s: a recall answer is one or two facts restated as a sentence, not a multi-source summary | `800` |
| `ENFORCE_ANSWER_LANGUAGE` | After answering, check the language deterministically and retry **once** if it is demonstrably wrong. Applies to `recall`, `review`, `research` and `sysadmin`. Never twice, never when either language is uncertain, and a failed retry keeps the first answer — wrong language with the right content beats an error message. The former name `RECALL_ENFORCE_LANGUAGE` is still read, so an existing `.env` keeps working | `true` |
| `MEMORY_RECALL_MAX_CHARS` | Cap on what a `memory` recall feeds back into the router prompt | `500` |
| `MEMORY_HARD_CAP_SLACK` | Headroom above `MEMORY_MAX_HISTORY` before the hard cap fires — sized so it fires rarely rather than every turn | `20` |
| `COMPACTION_TOKEN_THRESHOLD` | Prompt-token budget above which compaction triggers, alongside the message-count trigger (v3.12) | `6000` |
| `COMPACTION_TOKEN_TARGET` | What compaction aims to bring the history down to | `3000` |
| `EMBEDDING_MAX_CHARS` | Text longer than this is split before embedding, rather than truncated | `1500` |
| `EMBEDDING_MAX_CHUNKS` | Ceiling on chunks per embedded entry | `16` |
| `DELEGATE_EXECUTOR` | How a ready delegation job is executed. `handoff` writes the spec and stops there | `handoff` |
| `DELEGATE_DRAFT` | Ask the LLM to draft the spec before showing it. Off by default: on requests specific enough to act on, the draft invented detail the user never gave | `false` |
| `DELEGATE_ECHO_SECONDS` | Artificial delay in the echo executor, for testing the job lifecycle without a real handoff | `0` |
| `JOBS_FILE` | Persisted delegation jobs. Its own file rather than a key in `memory.json`: compaction rewrites that file wholesale, and two writers with one whole-file write means the job is what gets lost | `data/jobs.json` |
| `JOB_TIMEOUT` | Seconds before a running job is considered stuck | `1800` |
| `API_ALLOW_UNAUTHENTICATED` | Opt in to starting without an `API_TOKEN`. Refuses by default — the API dispatches tools on your machine | `false` |
| `API_DOCS_ENABLED` | Mount `/docs` and `/redoc`. Off by default | `false` |
| `ALLOW_MUTATION_AFTER_EXTERNAL_DATA` | Allow `shell`/`test`/`files:write` in the same run **after** external data was fetched. Off by default: the escalation guard is deterministic rather than asked of the model. `files:read` is deliberately non-tainting, to preserve the read-then-write flow | `false` |
| `POLICY_ALLOW_NETWORK` | Policy Engine: allow capabilities that reach the Internet (`research`, `web_fetch`, `web_search`, `shell`). A deny gate — it only subtracts from what `ENABLED_TOOLS` already permits, and a denied capability is not offered to the router at all | `true` |
| `POLICY_ALLOW_WORKSPACE_WRITES` | Policy Engine: allow capabilities that can write under `WORKSPACE_DIR` | `true` |
| `POLICY_ALLOW_SUBPROCESS` | Policy Engine: allow capabilities that spawn a process (`git`, `test`, `sysadmin`, `shell`) | `true` |

---

[← Documentation index](README.md) · [← Project README](../README.md)
