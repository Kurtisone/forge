# Configuration

Every setting lives in the environment. [`.env.example`](../.env.example)
is the exhaustive reference and carries the reasoning behind each value;
this page covers the ones you are likely to touch.

## The settings you are likely to touch

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
| `LLAMA_CPP_APPLY_TEMPLATE` | Wrap the prompt in the **loaded model's own** chat template, asked once from llama-server's `/apply-template` rather than written down here. Forge posts a raw prompt, so without this a model sees plain text where its training saw role markers — worth nothing on the model Forge was tuned around, worth 4 malformed replies in 36 on one whose template is ChatML. Off because a constant suffix after the growing prompt breaks the pure-append property (v3.12) for ~8 tokens a turn: cheap, not free, and unmeasured on a single-model deployment | `false` |
| `CAPABILITY_PROVIDER` | Which backend answers which capability, as `cap=provider` or `cap=provider:model`, comma-separated — e.g. `research=openrouter:z-ai/glm-4.6`. `FORGE_PROVIDER` is one value for the whole process; this makes the choice a property of the **work** instead. The router's own call is configurable as `router`. Not the Cognitive Scheduler and carrying no cost or quality scores: each capability still resolves to exactly one backend, from configuration. A malformed entry or an unknown backend is dropped with an error and the global one answers | *(empty)* |
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
| `SYSADMIN_USE_HARNAIS` | On a question naming **no** target, observe the machine through the Harnais (CPU, RAM, containers, kernel log) and hand the model a Context Builder text instead of a raw `journalctl -k` block. Every line it states is either a `[fact]` with a source and a timestamp or an `[unobserved]` saying what could not be read. Measured on with `bench/context_builder_ab.py` (Qwen3.8-9B, 2026-09-14): the log block loses two fixtures of four **outright** — a container restarted 247 s ago, and 94 % memory use — because `journalctl -k` cannot contain either, so resampling changes nothing; the control fixture is answered by both arms. Set `false` for the old path with no code change. A question that **names** a target is unaffected in either position, deliberately: asked about a container it could not observe, this model answered *"plante **car** le socket de Podman n'existe pas"* — the instrument's failure returned as the cause | `true` |
| `SYSADMIN_CONTEXT_BUDGET_TOKENS` | Token budget for the context built above. A ceiling that has not been touched: at this value no bench fixture reached it (the contexts came out around 800 tokens), so nothing was ever dropped and the truncation notice never fired | `1400` |
| `RECALL_MAX_ANSWER_CHARS` | Cap on a `recall` answer. Far smaller than `research`'s: a recall answer is one or two facts restated as a sentence, not a multi-source summary | `800` |
| `ENFORCE_ANSWER_LANGUAGE` | After answering, check the language deterministically and retry **once** if it is demonstrably wrong. Applies to `recall`, `review`, `research` and `sysadmin`. Never twice, never when either language is uncertain, and a failed retry keeps the first answer — wrong language with the right content beats an error message. The former name `RECALL_ENFORCE_LANGUAGE` is still read, so an existing `.env` keeps working | `true` |
| `RECALL_EXPANSION` | What to do when the distance cutoff drops everything, instead of refusing: `off`, `terms` (deterministic rewrites — **measured worse on this store, four questions out of four**, kept only so the finding stays reproducible) or `llm` (one model call asking for three questions rewritten with the words the *answer* would use). Runs on that path only, so it costs nothing on a question that already works and changes no distance `RECALL_MAX_DISTANCE` was calibrated against. **Inert with no cutoff set** — nothing is ever dropped, so there is never a failure to rescue; Forge says so at startup. **With one set, measured 2026-09-12: it fires on three questions of four, spends ~10 s each and returns nothing**, since the same cutoff drops its own rows — see [The measurement record](campaigns.md#the-hot-tier-and-what-it-subsumes-v318-concluded-2026-09-12). Measure it with `bench/in_container.sh recall_expansion` before turning it on | `off` |
| `RECALL_LEXICAL` | Search the **words** as well as the vectors (FTS5), unioned on entry id. Exists for entries no phrasing reaches — a fact stored as a keyword list is the shape an instruction-tuned embedding model retrieves worst. Word matches are ordered after measured distances and get their own budget, so a question that already works returns what it returned before with rows appended. **`RECALL_MAX_DISTANCE` is never applied to them**: that number was measured on vector distances and says nothing about a word match. Off until you have run `bench/in_container.sh rag_hybrid` on a copy of your own store — the cost is a row that shares a rare word with the question without answering it | `false` |
| `RECALL_LEXICAL_TOP_K` | How many word matches reach synthesis. Small on purpose: a recall prompt is paid for twice, in context window and in prefill time | `3` |
| `RECALL_LEXICAL_EXCLUDE_ARCHIVED` | Keep archived conversation out of the **word** channel only. An archived unit quotes the question verbatim (compaction indexes one exchange per entry), so for any question resembling one asked before, the transcript of that asking is the best word match in the store — and the emptier it is of answer, the better it matches. **Measured on the real store, 2026-08-25**: every junk row the word channel returned was archived transcript, both entries it rescued were facts. The vector channel still searches archives, so nothing becomes unreachable — only the circular route closes | `true` |
| `RECALL_HOT_FACTS` | Put the **whole** deliberate store (everything not archived transcript) into the recall synthesis prompt, unsearched, above the question. Not a fourth retrieval mechanism: the other three answer questions of *proximity*, this one answers a question of *completeness* by not asking a question at all — "list my hardware" wants a set, and a nearest-neighbour search returns the nearest rows of one however well calibrated. **Measured 2026-08-26**: the real store is 11 entries / ~195 tokens, prefilled at the 11.5–13.2 ms/token floor on *every* recall (~2.2–2.6 s), because both prompts share one llama-server slot and no prefix. Off until `bench/in_container.sh rag_hot_tier` has run on a copy of your own store | `false` |
| `RECALL_HOT_MAX_TOKENS` | Token budget for that block — a **tripwire, not a policy**. A cap forces a choice as soon as there are more entries than budget, and a choice is the ranking this tier exists to remove; 195 against 1000 is ~44 entries of headroom, and the aggregation tier lowers the count rather than raising it. Overflow cuts the **tail** (the only cut leaving survivors byte-identical) and says so inside the block: a silently short inventory reads complete and is wrong | `1000` |
| `RECALL_LEXICAL_MAX_DF` | The lexical channel's admission rule, and deliberately not a bm25 threshold — a score cutoff would be a second number needing its own calibration and its own tag. A word is searched for when it appears in at most this share of the store: `matériel` and `Podman` name a handful of rows, `mon` and `que` name half of it. The dictionary is the store itself, so it needs no stopword list and no French. A question of nothing but common words returns nothing from this channel, which is the point | `0.2` |
| `COMPACTION_AGGREGATE` | After a compaction, fold overlapping deliberate entries into one line per subject and point the sources at it (`superseded_by`). **No model call**: grouping is arithmetic on shared informative words, and so is the line — it is the sources' own details, verbatim, deduplicated. A 9B wrote it until 2026-09-11, when seven calls in two harness runs wrote zero aggregates; under a grammar that made repetition unsamplable it stopped repeating and started omitting. Off because of what it writes, not what it costs: an entry here is read as something you said, and it stays. `bench/in_container.sh rag_aggregate` shows every line it would write, on a copy, without writing | `false` |
| `COMPACTION_AGGREGATE_MAX_DF` | The share of the store above which a word identifies nothing, for naming subjects and for deciding which words a gate reads as carrying meaning. Starts at `RECALL_LEXICAL_MAX_DF`'s value because it is the same judgement about the same store, and is a separate knob because the two are measured against different things | `0.2` |
| `COMPACTION_AGGREGATE_MIN_SOURCES` | How many entries must fold before a line is written. Two is the floor that makes the word "aggregate" mean something: an entry standing in for one other entry is a rewrite of somebody's note. It does not apply to **absorption**, where one entry already carries every detail of another and speaks for it as it stands — nothing is rewritten there, and one source is enough | `2` |
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
| `FORGE_PUBLIC_URL` | Addresses the **phone** uses to reach Forge, **comma-separated**, in the order the app tries them — typically WireGuard first, then the LAN address. They travel verbatim inside the `!pair` QR code and become the client's base URLs, so a loopback address anywhere in the list builds a client that calls the phone itself — `!pair` refuses one. Unset, `!pair` refuses and says so | *(empty)* |
| `PAIRING_TTL_SECONDS` | How long a `!pair` code stays claimable. Also how long a photograph of the screen is worth anything | `300` |
| `ALLOW_MUTATION_AFTER_EXTERNAL_DATA` | Allow `shell`/`test`/`files:write` in the same run **after** external data was fetched. Off by default: the escalation guard is deterministic rather than asked of the model. `files:read` is deliberately non-tainting, to preserve the read-then-write flow | `false` |
| `POLICY_ALLOW_NETWORK` | Policy Engine: allow capabilities that reach the Internet (`research`, `web_fetch`, `web_search`, `shell`). A deny gate — it only subtracts from what `ENABLED_TOOLS` already permits, and a denied capability is not offered to the router at all | `true` |
| `POLICY_ALLOW_WORKSPACE_WRITES` | Policy Engine: allow capabilities that can write under `WORKSPACE_DIR` | `true` |
| `POLICY_ALLOW_SUBPROCESS` | Policy Engine: allow capabilities that spawn a process (`git`, `test`, `sysadmin`, `shell`) | `true` |

---

[← Documentation index](README.md) · [← Project README](../README.md)
