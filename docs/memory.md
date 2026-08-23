# Memory, RAG and traces

Forge has three distinct stores and they are easy to confuse: the
conversation history (one global file), the vector store (facts and
compaction pointers), and the execution traces (one line per run).

## Conversation Memory

Forge keeps a rolling history in `MEMORY_FILE`, capped by `MEMORY_MAX_HISTORY`, and injects
it as context into the router prompt on every turn. Every message carries a stable `id` and
a `pinned` flag.

Storage is plain JSON — no schema, no migrations, `cat data/memory.json` to inspect it.
Only genuine answers are persisted: a dispatch failure (`result.ok=False`) is never written,
and neither is a router-generated placeholder (empty/garbled model output, a detected repetition
loop, or leaked prompt instructions) — those succeed at dispatch (`chat` trivially echoes
whatever content it's given) but aren't real answers, and saving one as if it were would feed
it back into the next prompt as context, which can make a model that got confused once more
likely to get confused again on the very next turn.
Content is persisted in full — nothing is truncated on the way in, so a large tool result
(reading a whole file, for instance) shows up complete both in the router's own context and
in the web UI's `GET /history`, not cut short.

### Context compaction & drawer (v3.9)

`MEMORY_MAX_HISTORY` alone is a blunt instrument — a sliding window that drops the oldest
message every time a new one arrives once it's full, which also fights llama-server's prompt
cache reuse (v3.8) by shifting the whole history block on every eviction. `compaction.py` adds
a better mechanism ahead of that hard cap: once `COMPACTION_THRESHOLD` messages are reached,
the oldest non-pinned messages beyond `COMPACTION_KEEP_RECENT` are replaced by a single summary
message instead of being dropped outright. Two interchangeable strategies (`COMPACTION_STRATEGY`):
`rag_pointer` (default — no LLM call, pushes the block into vector memory verbatim and leaves a
short pointer, searchable via `!recall`/`/search`) or `llm_summary` (one LLM call, condenses the
block into prose kept inline). Both share the same signature, so switching is a config change,
not a rewrite. `POST /compact` (or `!compact` in the REPL) forces a pass on demand.

**What `rag_pointer` writes is one entry per exchange, not one per block.** Until
v3.14 it joined the whole evicted window into a single string and stored it as one
entry — and since that string is far past `EMBEDDING_MAX_CHARS`, its one vector was
the *average* of a dozen chunk vectors, which is close to no question in particular.
Measured against the real store on 2026-08-22: a question sat at 0.9386 from the
compacted block containing it nearly word for word, while a short fact sat at 0.671.
`forge/transcript.py` cuts the window at user turns (a user turn plus whatever
answered it is the smallest unit that still stands on its own) and `rag.remember_many`
stores the pieces as rows — the same number of embedding requests, kept apart instead
of averaged. `bench/rag_dilution.py` measures the difference; `deploy/rag_resplit.py`
re-slices blocks written before the change. Two things never reach the store: an
earlier compaction pointer (a reference to another entry, which answers no question)
and a raw router-JSON envelope (unwrapped to its content). Both paths go through
the same `transcript.units` / `transcript.split` pipeline, so a live eviction and a
migration cannot filter differently.

Measured on the real store, six questions before and after re-slicing: hits moved
from 0.8934 / 0.671 / 0.9386 to 0.4498 / 0.671 / 0.7695, turning a **NO GAP**
verdict (worst hit closer than the best miss) into a usable gap of 0.0642. The
short fact at 0.671 did not move — only the entries that had been buried did, which
is the control. That is what makes `RECALL_MAX_DISTANCE` a number one can choose;
see `.env.example` for the value and why it is not the harness's midpoint.

A message count turned out to be the wrong unit, though, so v3.12 added a
second trigger alongside it: compaction also fires when the rendered prompt
crosses `COMPACTION_TOKEN_THRESHOLD` tokens, aiming to bring it back to
`COMPACTION_TOKEN_TARGET`. Twenty short messages and twenty pasted stack
traces are the same number and nowhere near the same prompt, and it is the
prompt that costs. Either trigger can fire; whichever comes first wins.
`GET /context` reports what the next prompt currently weighs, which is what
the header gauge in the web UI reads.

Any message can be pinned — the "tiroir" (`GET /drawer`, `POST /drawer/pin`/`unpin`) — which
exempts it from both compaction and the `MEMORY_MAX_HISTORY` hard cap. The web UI pins a
question and its answer together by default (an answer read back without its question, or vice
versa, tends to lose its point), but either half can be unpinned independently afterward.

## Vector Memory / RAG (v3.7)

Separate from conversation memory above: a place to deliberately store decisions and
TODOs and retrieve them later by meaning, not just by recency. Backed by
[sqlite-vec](https://github.com/asg017/sqlite-vec), a single file (`RAG_DB_FILE`,
default `data/forge_rag.db`) with two tables — `memory_entries` (the actual rows) and
`memory_vectors` (a `vec0` virtual table), linked by `rowid`.

Two ways in:

```bash
# REPL — captures a decision/todo without leaving the session
Forge > !remember decision forge Use SQLite-vec instead of an external vector DB
Forge > !recall how should I index embeddings
  [decision/forge] Use SQLite-vec instead of an external vector DB  (distance=0.234)

# HTTP API — same auth/rate-limiting as every other route
curl -X POST http://localhost:8000/remember \
  -H "Authorization: Bearer $API_TOKEN" -H "Content-Type: application/json" \
  -d '{"kind": "decision", "content": "Use SQLite-vec", "project": "forge"}'

curl "http://localhost:8000/search?q=vector+db&top_k=5" \
  -H "Authorization: Bearer $API_TOKEN"
```

Embeddings are generated by a **separate, embedding-only llama.cpp instance**
(`EMBEDDING_URL`, default `http://127.0.0.1:8082/embedding`) — distinct from
`LLAMA_CPP_URL`, which stays dedicated to chat/tool-dispatch. This project uses
Qwen3-Embedding-0.6B (q8_0), served with `--embeddings --pooling last` (required for
this model: decoder-only, aggregates via the EOS token rather than mean/cls pooling)
and `--embd-normalize 2` (L2-normalized, so distance in `/search` is a plain cosine
similarity). `EMBEDDING_DIM` must match whatever model you actually serve — 1024 for
Qwen3-Embedding-0.6B.

**A third entry point, autonomous this time:** with `memory` in `ENABLED_TOOLS`, the
router itself can dispatch a `remember`/`recall` without a human typing a command —
"Remember that we decided X" or "What did we decide about Y" gets routed there like
any other tool, using the exact same `forge/rag.py` backend as the REPL commands and
the API. The prompt (`TOOL_DESCRIPTIONS["memory"]` in `router/prompt.py`) tells the
model to use it only on an explicit ask, matching how `files`/`shell`/`git` are
scoped — in practice a plain declarative statement ("I have a Steam Deck") gets
treated as an implicit remember too, which is closer to what a personal-assistant
usage pattern actually wants; tighten the wording there if you'd rather require an
explicit cue.

`kind` is `"decision"`, `"todo"`, or `"fact"` (a plain piece of information — the
one that matters for casual statements like the Steam Deck example above). If the
router's JSON omits `kind` entirely, the memory tool defaults to `"fact"` rather
than failing — a small local model asked to classify a plain statement on the fly
won't always supply one.

Recall's raw output is a bullet list (`- [fact] Possède un Steam Deck`), not a
sentence — same as `git`/`files` returning raw output directly. To get a natural
reply instead, the recall example in the router prompt sets `"done": false`, which
folds the raw result into history and lets the router run a second step to phrase
it as chat. **This requires `MAX_STEPS >= 2`** — at the default `MAX_STEPS=1` the
second step never runs and recall answers stay as a raw list, silently.

If the embedding server is unreachable, all three entry points fail the same
predictable way: `!remember`/`!recall` print a one-line error instead of crashing the
REPL, `/remember`/`/search` return `502`, and the `memory` tool returns a `[error]`
string the router treats as a normal (if unhelpful) tool result rather than a crash.

### What does not get indexed

Compaction indexes one entry per exchange — a user turn plus whatever answered it.
That means the question is *inside* the entry, and an exchange whose reply says
nothing is therefore a near-copy of its own question, which makes it the closest
possible match for anyone asking it again. **The emptier the entry, the better it
matches.** Measured on 2026-08-22: asked "Tu peux me lister mon matériel ?", the store
returned an archived refusal at distance `0.4519` ahead of the entry that actually
holds the hardware at `0.7891`.

Two filters keep those out, and they are deliberately different in kind:

- **the run says so.** A graph that ends without an answer reports it through
  `forge/outcome.py`, and the exchange is written to `memory.json` with
  `"answered": false`. Survives any change to the wording of the reply.
- **the text says so.** `forge/non_answer.py` holds the fixed strings Forge writes
  when it has nothing to say (`[error] `, `[no memory] `, `Tool error: `,
  `Something went wrong: ` and the cutoff refusal). This is the only test available
  to `rag_resplit`, whose input was written down long before any of this existed.

A recall answer is never indexed, good or bad. It was rebuilt from entries the
store already holds, so writing it back gives the store a second, worse copy —
worse because the copy carries the *question*, and an entry containing the
question outranks the entry containing the answer. Measured on 2026-08-23,
after the archived refusal was forgotten: `#138` came back at rank 1 for
"Tu peux me lister mon matériel ?" — and `#138` is itself an archived recall,
whose reply was already partial the day it was written. Left alone, every
recall adds one. The cost is that a good synthesised answer is not kept; it is
re-derivable from the entries it was built from, which are still there.

Either one drops the **whole** exchange, question included. Dropping only the reply
would leave an entry that is nothing but the question, which is the worst case rather
than a smaller one.

Neither catches a refusal the *model* phrased itself ("je n'ai pas cette
information") — that is prose like any other, and a phrase list aimed at it would
start dropping real answers. Failed turns stay in the conversation and on screen
either way; this only decides what the vector store is allowed to hold.

### Reading and repairing the store

`search` was the only reader this store ever had, and it is semantic by construction —
you cannot ask it what is *in* there without already having a question. `GET /memory`,
`!memory [kind]` in the REPL and `!memory` in the web UI list entries directly, with no
embedding call at all, and report the breakdown by `kind`. `!forget <id>` /
`DELETE /memory/{id}` remove one entry from both tables.

Two harnesses go with it, both writing to their own database and never to
`data/forge_rag.db`:

```bash
# what distance a good hit sits at, on this box, with this embedding model
python bench/recall_distance.py

# what burying a sentence in a compacted block costs
python bench/rag_dilution.py

# one-shot: re-slice blocks written before the per-exchange intake
python deploy/rag_resplit.py                      # dry run, the default
python deploy/rag_resplit.py --apply --backup /tmp/forge_rag.db.bak
```

A dry run prints every unit it would leave out as a non-answer, and lists the ids of
entries that turn out to be a refusal and nothing else. Those are reported, never
deleted — `!forget <id>` is the deliberate step.

`rag_resplit` rewrites rows in place and there is no undo — take the backup. It inserts
the pieces before deleting the block, so an interrupted run leaves a visible duplicate
rather than a missing entry.

## Execution Traces

Every run appends a record to `TRACE_FILE` (default: `data/traces.jsonl`):

```bash
tail -n1 data/traces.jsonl | python3 -m json.tool
# or inside the REPL:
!trace
# or via the API:
GET /traces?n=5
```

Each record contains: `run_id`, `timestamp`, `user_input_preview`, per-step tool + duration,
`total_ms`, `ok`, `error`.

---

[← Documentation index](README.md) · [← Project README](../README.md)
