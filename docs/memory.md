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

### The query instruction

Qwen3-Embedding is instruction-aware, and its retrieval format is **asymmetric**:
the instruction goes on the query, the document is embedded raw. Forge applies it
in `rag.search` and nowhere else — documents reach the store through `remember`,
queries through `search`, so the asymmetry is structural rather than a rule
someone has to remember. It is also why turning it on needed no migration: every
vector already stored was written raw and stays valid.

Measured against the real store on 2026-08-23, three hits and three misses, the
same six questions raw and prefixed (`bench/instruct_prefix.py`):

| | worst hit | best miss | gap |
|---|---|---|---|
| raw | 0.8366 | 0.8337 | **−0.0029** |
| prefixed | 0.8951 | 0.9495 | **+0.0544** |

A negative gap means no threshold exists at all. What makes the result
trustworthy is not the size of the number but that all six questions moved the
way one mechanism predicts: **the prefix pulls weight off literal string
overlap.** The semantic hit improved by 0.108; the two that got worse are the two
that were scoring on overlap, and one of them is a miss that was supposed to move
away.

Read those two rows with one caveat: at the time they were produced, the harness
scored a hit on whatever row came back first rather than on the entry `--expect`
named, so a question whose answer was outside the top 5 contributed the distance
to something else. The mechanism story is unaffected — it rests on the direction
each question moved, not on the size of the gap — but the **+0.0544** is not a
number to calibrate anything against. Re-run `bench/instruct_prefix.py` for that;
it reads the named entry now.

Set `EMBEDDING_QUERY_INSTRUCT=` (empty) for an embedding model that is not
instruction-aware — for those it is noise glued to every search.

### A threshold belongs to the regime it was measured in

`RECALL_MAX_DISTANCE` may carry the embedding configuration it was calibrated
against, as `0.95@e3b0c4`, where the tag is `rag.query_fingerprint()` — six hex
characters over the whole query wrapper, so changing the instruction *or* the
`Instruct:/Query:` shape around it is a change of regime.

| tag | what happens |
|---|---|
| matches | used, silently |
| missing | used, with a startup warning naming the fingerprint to write back |
| stale | **cutoff off**, loudly |

Off rather than adjusted: a number from another regime is not too high or too
low, it is unrelated, and the two ways of being wrong are not symmetric. Too high
lets a bad answer through and a bad answer gets argued with; too low answers "je
n'ai rien en mémoire" while the entry is sitting in the store, and that gets
believed.

This is not a hypothetical rule. The 0.95 that used to be in `.env.example` was
measured on raw queries, the query instruction shipped on by default the next day,
and in the new regime the best miss came back at 0.9495 — *under* the cutoff. The
filter that had been validated in real use had quietly stopped cutting the case it
was validated on.

Re-measured 2026-08-24 in the regime that actually ships — six real questions,
known answers, hits scored on the entry `--expect` names:

| | distance |
|---|---|
| hits | 0.7289 (`#308`), 0.6640 (`#17`), 0.6469 (`#309`) |
| misses | 0.9495, 1.0451, 1.1578 |
| gap | **0.2206** |

`.env.example` ships `0.88@a5c47b` from that run: above the 0.8392 midpoint,
because real entries are longer and messier than fixtures and the room belongs
above the hits.

**One thing the tag cannot tell you.** It catches a change of embedding
configuration. It does not catch the number having been measured on somebody
else's store — keep the default instruction and your fingerprint matches the one
that ships, so an inherited 0.88 engages silently on data it has never seen. A
store of a different size, language or subject puts its hits somewhere else.
Comment the line out for no filtering at all, which is the safe starting point,
and run the harness against a copy of your own store to earn a number of your
own.

The limit, stated plainly: this fingerprints what Forge controls. Swapping the
embedding model behind the same `EMBEDDING_URL` moves every distance in the store
and leaves the tag identical.

### When the question is the problem, not the threshold

Measured on the real store, 2026-08-24, both questions about the same hardware:

| question | result |
|---|---|
| `Quel processeur a mon NiPoGi ?` | `#308` at rank 1, 0.7289 |
| `Tu peux me lister mon matériel ?` | `#308` nowhere in the top 5 |

No value of `RECALL_MAX_DISTANCE` separates those, because the entry never comes
back to be filtered. Retrieval here follows the wording as much as the meaning,
and a note reading *processeur Ryzen 5500U, 32 Go de RAM* does not share a word
with a question that says *matériel*.

So `RECALL_EXPANSION` asks again, in other words — **on the query, never on the
stored fact**. A wrong expansion of a query searches somewhere useless and the
cutoff throws the result away; a wrong expansion of a fact writes something
nobody said into memory, where it is indistinguishable from something they did,
and it stays.

| mode | what it does |
|---|---|
| `off` | nothing. The default. |
| `terms` | strips the conversational frame and the stopwords. **Measured worse, four questions out of four** — see below. Kept so the finding stays reproducible; do not turn it on. |
| `llm` | one model call under its own grammar, asking for three **questions** rewritten with the words **the answer** would use. |

### Measured, and shipped off

**Verdict, 2026-08-24: `llm` works as a mechanism and there is no threshold to
give it on this store. It ships `off`.** Six rounds of measurement, three
readings of the gap between what the rescue finds and what it drags in, and the
gap came out negative every time.

The mechanism is not what failed. It moves the right entries closer — `#307`
went from absent-from-the-top-5 to rank 1 — and pushes the wrong ones away: the
cat question receded from 0.9495 to 1.1263, the tarte tatin and the social
security number stopped returning anything at all. What it cannot do is
*separate*. On a store holding five or six short facts that all overlap, every
distance bunches between 0.94 and 1.14, answers and non-answers alike, and the
closest competitor to the hardware question sits 0.009 from the answer.

Every failure traced back to the store, not to the search:

| what went wrong | what it really was |
|---|---|
| `#167` and `#176` beat every fact on technical questions | archived transcripts holding tool output — long, noun-dense, unbeatable by a one-line fact. Now excluded from the rescue. |
| two questions of three had no answer at all | ~290 archived exchanges, most of them refusals, against five facts |
| `#315` unreachable by any rewrite | stored as `NiPoGi AM06PRO, Arch, 5500U, 32Go RAM` — the words a question would use were dropped at write time |
| category rewrites lost to brand-name rewrites | the facts are themselves bags of product names, so only product-name queries match them |

That last row is the one to sit with. Forbidding the model to guess a brand made
retrieval *worse* here — `#313` went from 0.9941 to 1.1341 — because `#313` is
`Steam Deck, SteamOS, conteneurs Podman`, a telegram of product names. The rule
is still right: a guessed brand is a wrong answer waiting to happen on any store
whose owner runs something else. It looks wrong here only because the write path
had already reduced the facts to the same shape as a bad query.

**So the order of work is: fill the store, then calibrate the rescue.** Turning
`RECALL_EXPANSION=llm` on today buys a 7-second model call on every refused
question and rescues nothing. The harness is here, the numbers are here, and the
question can be reopened in one command the day the store has enough facts to
tell an answer from a neighbour.

Measured against the real store, 2026-08-24, distance to the named entry (or to
the closest row where it was absent):

| question | baseline | `terms` | `llm` |
|---|---|---|---|
| `Tu peux me lister mon matériel ?` | 0.9083 *(#308 absent)* | 1.0277 | **0.9766** *(#308 at rank 2)* |
| `Combien de RAM a le NiPoGi ?` | 0.7336 | 0.8591 | — *(already within the cutoff)* |
| `Comment s'appelle mon chat ?` *(miss)* | 0.9495 | 1.0821 | 1.0431 |

`terms` lost on every question it was asked, hits and misses alike. The embedding
model is instruction-tuned on natural-language queries, and a keyword bag is
off-distribution for it — even the mild rewrite, still a phrase, lost by more
than a tenth. What worked was the model's rewrites, and those are *phrases in the
store's own vocabulary* (`matériel ordinateur portable`, `processeur mémoire
disque`), not the question with its function words removed. Stripping words from
a question does not make it a better query here; it makes it a worse sentence.

### The rewrites are questions, and that is a grammar rule

The second pass of 2026-08-24 ran `llm` again on the real store and found the
mechanism sorting correctly and scoring worse. `#307` and `#313` both came back
at **rank 1**, and both *further away* than the baseline they replaced —
0.9083 → 0.9488 and 1.0400 → 1.1341. The cutoff filters on distance, not on
rank, so every rescue landed above `0.88` and the pass rescued nothing.

The variants said why: `['matériel ordinateur', 'équipement informatique',
'configuration système']`. Keyword bags — which is what the prompt had asked
for, in as many words: *"no question mark, no politeness — these are search
queries, not questions."*

So `terms` versus `llm` was never the comparison. The keyword **shape** lost
twice: once written by a regex, once by a model told to write one. The
vocabulary bridging is the half that worked; the shape it arrived in is what
cost the distance, on an embedding model instruction-tuned for natural-language
queries.

The rewrites are complete questions now, in the answer's vocabulary, and the
grammar is what enforces it — `string ::= "\"" word (" " word)+ " "? "?" "\""`,
with `?` excluded from `word`. Not the prompt: a rule the model is *asked* to
follow is one it follows most of the time, and the times it does not are the
ones nobody sees. That makes ten occasions on this repository where a
deterministic constraint replaced a phrasing that had been trusted. A rewrite
without a question mark is **dropped, never repaired** — punctuating a keyword
bag produces the losing distances under a passing check.

Re-measurement pending; the numbers above are the keyword-form pass and the
comparison to beat.

The model call costs ~7 s on the Deck for a ~330-token prompt — against the ~47 s
the router already spends, and only on the path that was going to refuse.

It runs on **one path**: after the cutoff has dropped everything, instead of "je
n'ai rien d'assez proche". Three things follow.

- **Free on every question that already works.** A recall whose first pass keeps
  something never builds a variant and never spends the model call.
- **It cannot displace a good answer.** The alternative outcome on that path is a
  refusal.
- **It changes no distance the threshold was calibrated against.** The first pass
  is untouched, every variant goes through the same query wrapper, and the merge
  keeps real query-to-entry distances — the minimum across phrasings, never a
  statistic over several. `0.88@a5c47b` and its tag stay valid; nothing has to be
  re-measured before this can be used.

**With no cutoff set it can never fire** — nothing is ever dropped, so there is
never a failure to rescue. Forge says so at startup rather than leaving it to be
discovered.

What it *can* do is let a miss in: a rephrasing that sits nearer some unrelated
entry than the question did. That is the price, and `recall.rescued` logs every
one with the entry id, the distance and the **variant** that found it, because
when a rescued answer turns out to be wrong the question is which rephrasing
dragged it in. The sub-trace says *après reformulation* for the same reason.
`bench/in_container.sh recall_expansion` counts both sides on a copy of the real
store; it is what earns the setting.

### Reading and repairing the store

`search` was the only reader this store ever had, and it is semantic by construction —
you cannot ask it what is *in* there without already having a question. `GET /memory`,
`!memory [kind]` in the REPL and `!memory` in the web UI list entries directly, with no
embedding call at all, and report the breakdown by `kind`. `!forget <id>` /
`DELETE /memory/{id}` remove one entry from both tables.

Four harnesses go with it, none of which ever writes to
`data/forge_rag.db`. `bench/in_container.sh` copies the checkout and a fresh copy
of the store into the container and runs one of them there — it is the six-command
`podman cp` sequence that used to sit at the top of each file, where forgetting a
line was silent:

```bash
# what distance a good hit sits at, on this box, with this embedding model
bench/in_container.sh recall_distance

# whether the query instruction helps on this store (needs --expect ids)
bench/in_container.sh instruct_prefix --db /tmp/real_copy.db \
    --hit "Quel processeur a mon NiPoGi ?" --expect 308 \
    --miss "Comment s'appelle mon chat ?"

# what asking again in other words rescues, and what it lets in
bench/in_container.sh recall_expansion --db /tmp/real_copy.db \
    --hit "Tu peux me lister mon matériel ?" --expect 308 \
    --miss "Comment s'appelle mon chat ?"

# what burying a sentence in a compacted block costs
bench/in_container.sh rag_dilution

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
