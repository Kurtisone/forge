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

- **the run says so.** A graph node whose whole purpose is to report that Forge
  has nothing is declared `answers=False` where it is registered, and `Graph.run`
  reports it through `forge/outcome.py`; the exchange is then written to
  `memory.json` with `"answered": false`. Survives any change to the wording of
  the reply.
- **the text says so.** `forge/non_answer.py` holds the fixed strings Forge writes
  when it has nothing to say (`[error] `, `[no memory] `, `Tool error: `,
  `Something went wrong: `, `[no results] `, sysadmin's `[cible introuvable] ` and
  `[collecte impossible] `, the delegation flow re-asking, and the cutoff refusal).
  This is the only test available to `rag_resplit`, whose input was written down
  long before any of this existed.

### Both filters were narrower than they read, measured 2026-09-12

195 archived entries in the real store, **22 of them the assistant refusing**, and
`non_answer` recognised **none**. The two filters are sound and neither covered
what it appears to cover.

The text filter is a closed set with no way to discover its own members: four
producers had never been registered — the research graph and the `web_search`
tool on an empty search, sysadmin's two nodes written entirely in code, and the
delegation flow re-asking mid-job. Registering them (and making every producer
import the constant, which is that module's stated anti-drift rule) recognises
**13 of the 195**, every one of them a genuine refusal and no real answer among
them. `deploy/rag_resplit.py` now lists those thirteen ids for `!forget`.

The run filter had been wired into one graph. `recall` was told to report itself
in August; `research`, `review`, `sysadmin` and the default fallback were not —
and each of them sets `ok = True` in its refusal node for the same good reason,
so the user reads a message rather than a crash, which is exactly what erases the
fact the store needs. It is a declaration on the node now, not a call inside it.

**What still gets through, and it is the interesting half.** Nine of the
twenty-two are prose a model chose, and they split in two. Five are a chat turn
declining ("Je ne peux pas analyser les logs de ton Steam Deck") — an answer like
any other, and a phrase list aimed at it would start dropping real answers. Four
are a `sysadmin` or `research` **synthesis** reporting that the logs it collected
or the results it found do not answer the question. Those looked different: the
run has the material to know, it ran the collection, and nothing in it reports a
verdict — the only trace is a sentence a model chose to write. (The other four of
the twenty-two were recalls, which stopped being indexed at all on 2026-08-23.)

### The verdict the code could read, asked for four ways

Giving that synthesis a verdict instead of a sentence was the obvious next
mechanism, and `bench/sysadmin_verdict.py` is what it had to get past: eight
fixtures whose answer is known, one model call each, on 2026-09-12.

| arm | right | FALSE NO | FALSE YES |
|---|---|---|---|
| `plain` — do these logs answer the question? | 5 | 0 | 3 |
| `negated` — is the answer missing from them? | 5 | 3 | 0 |
| `cite` — which line answers it, or NONE | 5 | 0 | 3 |
| `reason` — one bounded phrase, then the verdict | 3 | 3 | 2 |

**Read the two error columns and ignore the first one.** `plain` is wrong only
in the YES direction and `negated`, which is the same question inverted, only in
the NO direction: each arm answers the *shape* of the question it was asked and
scores on exactly the fixtures where that shape happens to be right. `cite` could
have said NONE and never did where it had the option — line 1, six times out of
seven. `reason` filled its phrase slot with hallucinated prompt instructions
("Keep the answer short. No explanations.") in half the cases, and on the one
where it described the crash correctly it then answered NO.

So there is no verdict channel at this model size, in any of the four shapes.
What the campaign did produce is the case that never needed a model: `plain` said
an **empty** log block contained what was needed to answer a question about a
restart, and `graphs/sysadmin.py` now refuses that in code, with
`[rien à lire] ` and a node declared `answers=False`. An empty log file is not a
quiet system.

The rest of the family stays open, and the harness is the cheapest thing in this
repository to re-run the day a larger provider is wired up.

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

And both arrive late, always. The mark is applied when the exchange is persisted
and read when compaction fires, so **any rule about what compaction may index has
a latency equal to the lifetime of the rolling history** — a filter merged today
protects nothing already sitting unmarked in `memory.json`, and nothing at all
already in the store. That is what `!forget` and `rag_resplit`'s report are for.

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

## Two channels, two admission rules (v3.17)

Six rounds of query expansion ended in a finding no threshold answers: **`#307`
and `#313` are in the store and no natural-language question reaches either
one.** `#313` is stored as `Steam Deck, SteamOS, conteneurs Podman` — a fact
written as a keyword list, which is the shape an instruction-tuned embedding
model retrieves worst. Rephrasing it, expanding it, moving the cutoff: none of
them can work, because the entry never comes back to be filtered. The words are
sitting in the entry and nothing was looking for them.

So recall searches the words too. `memory_fts` is an FTS5 index over the same
rows — external content, so the text exists once and the two tables cannot
disagree about what an entry says — kept in sync by triggers rather than by
calls in `_insert`/`forget`, because four paths write this store and three
delete from it, including a human with `sqlite3` open. FTS5's own `rebuild` is
the migration: the index is created and backfilled on the first connection
after the upgrade, and only then. A SQLite built without FTS5 warns and keeps
the vector channel.

### The rule of the word channel is on the query side

`RECALL_MAX_DISTANCE` is a number measured against one embedding configuration
and tagged with it, because a distance means nothing outside its regime. bm25 is
worse on that axis, not better: it is a score relative to a corpus, so a cutoff
on it would need its own calibration, its own tag, and a re-measurement every
time the store grows.

This channel therefore admits on the **query** instead. A word earns a place in
the search when it appears in few enough entries to separate them —
`RECALL_LEXICAL_MAX_DF`, a share of the store, `0.2` to start.
`matériel`, `Podman`, `5500U` name a handful of rows; `mon`, `peux`, `que` name
half the store, and searching for those returns the store. The dictionary is the
store itself, the same choice already made for the spelling check on writes: a
stopword list is a maintained artefact that is wrong for whatever gets written
next, while a frequency count over the actual entries is right by construction,
sharpens as they accumulate, and needs no French — which matters for a corpus
holding NiPoGi, busctl and aardvark-dns.

A question made entirely of common words returns **nothing** from this channel.
That is the point rather than a gap: matching on `mon` because it was the only
word left hands synthesis a random slice of the store carrying the authority of
a lexical hit.

Nothing in that judgement touches the embedding model, so **`0.88@a5c47b` is
untouched and was not re-measured**.

### The union, and where the cutoff stops

`rag.search_hybrid` runs both and merges on entry id. A union, not an
intersection: the two channels fail on different things — the vector channel
cannot reach an entry that is not written in prose, the lexical channel cannot
reach an entry that answers in other words than the ones asked — so an entry
either one finds is an entry Forge can answer from. Requiring both would keep
only the questions that already worked.

No fused score. Normalising a distance and a bm25 into one number, weighting
them, reciprocal-rank fusion: every one of those puts a tunable constant between
the store and the answer, and every tunable constant in this file has cost a
measurement campaign to place. Measured distances come first, word matches
follow, each ordered by its own number. Each channel keeps its own budget rather
than sharing `top_k`, because a shared one lets five vector rows crowd the word
matches out — and the case this exists for is exactly the one where those five
are about to be cut.

Every row carries `channel` (`vector`, `lexical` or `both`), and **the distance
cutoff is applied to vector rows and to nothing else.** A row the word channel
admitted was chosen because the question and the entry share words rare enough
to mean something, which is a different question with a different answer;
judging it by a distance would not be strict, it would be a category error — and
it would delete precisely what the second channel was added to reach, since
`#307` and `#313` are *far* in vector space. A `both` row keeps its distance for
the log and for ordering, and stops being sentenced by it.

### Measured on the real store, 2026-08-25

193 entries, three questions, three values of `RECALL_LEXICAL_MAX_DF`:

- **`#307` and `#313` are reached.** Neither is reachable by the vector channel at
  all — the hardware question's nearest row was `#176` at `0.9083` and the
  container question's four nearest were all past `1.00`. This is what the branch
  was opened for, and it works.
- **`MAX_DF` is not the lever.** `0.05`, `0.2` and `0.5` returned the *same three
  rows* on the miss, byte for byte, and only changed bm25 magnitudes on the hit.
  A knob that moves numbers without separating anything is the shape the
  expansion campaign spent six rounds establishing about a different knob.
- **Everything the word channel dragged in was archived transcript**: `#94`,
  `#61`, `#108`, `#273`, `#49`, `#70`, `#75`. Everything it rescued was a `fact`.

That last line is not a coincidence, it is the shape of the data. An archived
unit **contains the question**, verbatim, because compaction indexes one exchange
per entry. So for any question resembling one asked before, the transcript of
that asking is the best word match in the store — and the emptier it is of
answer, the better it matches, since bm25 rewards the query terms being a large
share of a short document. `#94` is `Tu peux analyser les logs de mon Steam
Deck ? / Je ne peux pas…`: a refusal that outscored the hardware fact on the
hardware question.

This is the **third arrival of one finding by a third route** — `#272` in
`feat/rag-non-answers`, the intake measurement above, and now this. On the word
channel it is not a bias but circularity: matching a question against a copy of
itself. Hence `RECALL_LEXICAL_EXCLUDE_ARCHIVED`, on by default, which closes that
route and nothing else — the vector channel keeps archived conversation in scope,
so `0.88@a5c47b` keeps the scope it was measured in and no entry becomes
unreachable. `--include-archived` measures the other side in one command.

### What it still cannot do, from the same run

`Tu peux me lister mon matériel s'il te plaît ?` came back with `#307` and
nothing else — the whole answer, correctly, out of one entry. The Steam Deck was
not in it, and the reason is the limit this page already states from the other
side: **the word channel cannot reach an entry that answers in other words than
the ones asked.** `#313` is `Steam Deck, SteamOS, conteneurs Podman`; it does not
contain `matériel`, so the words miss it, and the vector channel cannot reach it
either. Both channels fail on that entry for that question, for opposite reasons,
and the union of two channels is not a third one.

Two consequences worth keeping.

**"A fact should name its category" is un-falsified for the word channel.** This
page records that rule being proposed, measured and abandoned — a fact carrying
the word `matériel` sat at rank 109 for `Tu peux me lister mon matériel ?`. That
measurement was of the *embedding*, and it stands. bm25 has no such problem:
`#307` came back at rank 1 on 2026-08-25 **because** it contains the word. The
rule is not resurrected in general, it is resurrected for the channel that reads
words — which is worth having in hand when the fact-extraction lot opens, and
worth not confusing with the finding that killed it.

**Retrieval answers proximity questions, not completeness questions.** "List my
X" wants a *set*; both channels return the nearest rows, and `RECALL_LEXICAL_TOP_K`
is a hard ceiling on how many a list-shaped question can gather. Spreading one
subject across entries that share no vocabulary makes it unlistable by any
mechanism here.

There is also no floor under bm25: on `Sur quoi tournent mes conteneurs ?`,
`#313` came back at `-6.43` followed by `#10` and `#309` at `-2.29`, admitted
only because the budget allowed three rows. Harmless there — both are short facts
— but the channel always spends its whole budget, and a relative floor would be
the second calibrated number this design has so far done without.

### The cost, which is the same shape as last time

A row that shares a rare word with a question without answering it. This store
is full of the family: the archived refusals `#35`/`#36`/`#37` **quote** the
question they failed to answer, which makes them excellent lexical matches for
it. That is why `RECALL_LEXICAL` ships `off` and why the harness counts both
sides:

```bash
bench/in_container.sh rag_hybrid --db /tmp/real_copy.db \
    --hit "Tu peux me lister mon matériel ?" --expect 307 \
    --hit "Sur quoi tournent mes conteneurs ?" --expect 313 \
    --miss "Comment s'appelle mon chat ?"
```

`REACHED` is what the word channel returns and the vector channel does not
*deliver* — absent from its top-k, or inside it and beyond the cutoff. Coming
back is not reaching synthesis. `INTRUDERS` is a `--miss` the word channel
answers: a correct refusal turning into a fluent wrong sentence, the cost the
expansion pass was measured on and turned off for. `WRONG ENTRY` is reached with
something else ahead of it. `--max-df` is repeatable, one column per value;
`--no-vector` runs the whole thing with the embedding server down and refuses to
claim a rescue it never measured.

There is deliberately **no suggested threshold** in that output. Producing one
would invent the second calibrated number this channel was designed not to need.

## The hot tier (v3.18)

Six campaigns end on one sentence, and this page states it twice from
opposite sides: **retrieval answers proximity questions, not
completeness questions.** `Tu peux me lister mon matériel ?` wants a
set. A threshold, a rephrasing and a second channel each improved the
ranking, and v3.17 proved the point from the good side — it reached
`#307` and `#313`, which nothing had reached before, and still
answered out of a single entry.

So for the entries small enough to fit, recall stops searching and
looks. `rag.hot_entries` reads everything that is not
`ARCHIVED_KIND`, oldest first, and `forge/hot_memory.py` renders it
into the synthesis prompt whole.

Measured on the real store, 2026-08-26:

| | |
|---|---|
| deliberate entries | 11 |
| the block | 700 characters, ~195 tokens |
| exact duplicates | 0 |
| synthesis prompt it joins | ~1134 tokens |

**Not `kind = 'fact'`.** This page records the memory tool defaulting
a missing kind to `fact` rather than failing, so the kind on any row
the router wrote is a 9B's guess made on the fly. What is reliable is
the binary distinction `memory._rank` already sorts on: someone chose
to write this down, or compaction dumped it here. The `decision`
about emoji pinning is in the block for that reason.

### The cap is a tripwire, not a policy

A token budget forces a choice as soon as there are more entries than
budget, and a choice is a ranking — precisely what this tier removes.
Three ways out were on the table: drop by age, let the user pin, or
keep the count small enough that the question does not arise. The
measurement settles it. 195 against `RECALL_HOT_MAX_TOKENS=1000` is
five times the headroom, and the aggregation tier that comes next
lowers the count rather than raising it.

When it does trip, it cuts the **tail** — the one cut that leaves
every surviving line where it was — and says so inside the block. A
silently short inventory reads complete and is wrong, which is
strictly worse than the visibly incomplete answer this replaces, and
it is the only way this mechanism can make Forge worse than not
having it.

### Above the question, and why that is the whole placement argument

The synthesis prompt is ordered preamble → question → retrieved
entries. Anything spliced into the entries block therefore sits behind
a string that changes every turn, so a block byte-identical from one
recall to the next would still never be a shared prefix down there.

**And it is one, measured.** The branch shipped believing otherwise —
the argument was that both prompts share one slot
(`LLAMA_CPP_ID_SLOT`) and share no prefix, so the synthesis prompt
would be recomputed whole every time. Three consecutive real runs on
2026-08-26 say no:

| run | prompt_n | prompt_ms | ms/token |
|---|---|---|---|
| 1 (cold) | 702 | 7720 | **11.0** |
| 2 | 661 | 5295 | **8.01** |
| 3 | 718 | 5789 | **8.06** |

At the cold floor of 11.0, runs 2 and 3 re-evaluate ~481 and ~526
tokens — leaving ~180 and ~192 never recomputed, against a block of
195. And a router call landing *between* two synthesis calls came back
at `prompt_n=5755, 0.15 ms/token`: the intervening call had not
evicted it. This build keeps more than one sequence.

So the block costs **~2.2 s once**, not per recall, and the cap has far
more room than it was given. The placement above the question is what
earns that; below it, the block would sit behind a string that changes
every turn and would be recomputed every time whatever the slot did.

The framing has to establish two things, and the second is not
obvious. That the list is **complete**, or the model hedges a complete
answer. And that it is **description, not instruction**: the store
holds `[decision/chat preferences] Ne pas épingler les messages avec
des emojis`, an imperative sentence about Forge's own behaviour, now
sitting at the top of every synthesis prompt.

### Nothing close enough is not nothing to answer from

`_recall_node` short-circuits to the error node when the cutoff drops
every row, with no LLM call — 4 s instead of 17, and that is right.
It is also the path `Tu peux me lister mon matériel ?` takes, which is
the question this tier exists for. Left alone, the block would have
been invisible in exactly the case it was built for.

Both refusals therefore gain the same second half: the run continues
if there is a block, whatever the search returned. The **embedding
outage** path deliberately does not, though it could — reading the
block is a plain sqlite `SELECT` and needs no embedding server. All
three entry points of this store fail the same predictable way on
purpose, and answering fluently from a partial capability is how an
outage lasts a week.

### What five real runs established

The mechanism works and the branch does not solve the question it was
opened for. Both halves matter.

**Validated, in the sharpest possible form.** On `Quels sont tous les
ordinateurs que je possède ?` the store returned nothing at all —
`#1` at 0.959, `#2` at 0.967, `#307` at 1.023, `results=0` after both
the vector channel and the expansion pass — and the answer was
complete and correct: the Dell R710, the NiPoGi AM06PRO and the Steam
Deck, three for three, picked out of eleven lines of which eight are
not computers. Retrieval contributed *nothing* and the answer was
right. That is the thesis of this tier demonstrated from the inside.

**The ranking did not disappear, it moved.** On `Tu peux me lister mon
matériel ?` the same block produced one entry, then two, then one
again across three runs. A model that sees eleven lines and returns
one is not failing to enumerate — the computers question proves it
enumerates and filters by sense without dropping anything. It is
judging **scope**: "mon matériel" read as one machine rather than
four, which is a defensible reading given three overlapping NiPoGi
entries.

So the correct next move is **not** a grammar and **not** more words
in the prompt. GBNF was the candidate structural fix and the computers
run closes it. Overlapping entries are the aggregation tier's work,
and this is now the second independent reason to build it.

### The store poisons itself, and the moment is compaction

Four archived transcripts of `Tu peux me lister mon matériel ?` sat at
0.5757–0.7312 — closer than any fact has ever come to that question —
and three of them were refusals. One of them, `#329`, was a previous
trial's own answer, which had become the best match for the next
trial. A diagnostic run reading them answered by copying `#329`'s
opening words.

Asking the question is not what does this. Compaction is:
`compaction.indexed messages=59 … entries=17 first_id=318 last_id=334`
wrote seventeen entries in one pass, four of them transcripts of the
test question. Between the trial and the threshold there is a window
in which nothing has been indexed yet, and `!clear` inside it costs
nothing where cleaning up afterwards costs a probe, four `DELETE`s and
a store that served false results in between.

Two consequences. Real trials need the same discipline `bench/`
already has, and the "measure on a copy" rule never covered them.
And `!clear` currently wipes **pinned** messages too, so the cheap
window is only cheap for someone with nothing pinned — a drawer that
does not distinguish what was deliberately kept is the same design
fault this tier was built to fix in the store.

### What this does to the word channel

`RECALL_LEXICAL_EXCLUDE_ARCHIVED` ships `true`, so the word channel
only ever returns non-archived rows — which is exactly the set the hot
block now carries whole. `#307` and `#313`, the two entries v3.17 was
opened for, are both in it.

That is a claim about a branch merged the day before, so it gets
measured rather than asserted. `bench/rag_hot_tier.py` reports
**SUBSUMED** (rows the word channel returns that are already in the
block) against **ADDS** (rows it returns that are not), plus the cost
in prefill seconds and the headroom under the cap:

```bash
bench/in_container.sh rag_hot_tier --db /tmp/real_copy.db \
    --hit "Tu peux me lister mon matériel ?" --expect 307 \
    --hit "Sur quoi tournent mes conteneurs ?" --expect 313 \
    --miss "Comment s'appelle mon chat ?"
```

Measured on the real store, 2026-08-26: **`SUBSUMED 4 / ADDS 0`** —
every row the word channel returned was already in the prompt. With
`--include-archived` it becomes `SUBSUMED 1 / ADDS 2`, and the two it
adds are `#94` and `#61`, archived transcripts, which is exactly what
`RECALL_LEXICAL_EXCLUDE_ARCHIVED` exists to keep out.

**Measured, not concluded.** The redundancy is real and the decision
is not taken: v3.17 was measured and won on its own terms, and turning
it off on the strength of numbers collected during a diagnostic of
something else is the mistake this page records about few-shot
reasoning. The same holds for the expansion pass, which spent 9.3-9.5 s
returning nothing on two questions whose answer was already in the
prompt. Both are the same open question — two rescue mechanisms whose
useful share the hot block absorbs — and they get measured together,
for themselves, or not at all.

There is deliberately **no suggested budget** in that output. The cap
is a budget and not a measurement; the only thing that moves it is the
store growing, and `HEADROOM` reports that directly.

### What it does not do

It does not aggregate. Three entries in the real block overlap without
any pair being identical:

```
- [fact] Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go
- [fact] NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible, services Podman
- [fact] Le NiPoGi a 32 Go de RAM
```

No deterministic test merges those, and exact-duplicate detection
finds nothing — there are zero exact duplicates in this store. That is
the next tier's work, and the block is now the place it is legible.

It does not scale past the cap either. When the deliberate store
outgrows the budget the answer is an index and sub-documents loaded on
demand, as a graph, and not a bigger block.

### Reading and repairing the store

`search` was the only reader this store ever had, and it is semantic by construction —
you cannot ask it what is *in* there without already having a question. `GET /memory`,
`!memory [kind]` in the REPL and `!memory` in the web UI list entries directly, with no
embedding call at all, and report the breakdown by `kind`. `!forget <id>` /
`DELETE /memory/{id}` remove one entry from both tables.

Five harnesses go with it, none of which ever writes to
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

# which channel reaches which entry, and what the words drag in
bench/in_container.sh rag_hybrid --db /tmp/real_copy.db \
    --hit "Tu peux me lister mon matériel ?" --expect 307 \
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

One harness in `bench/` reads no store at all: `sysadmin_verdict` asks whether the
model can tell that a log block does not answer a question, over fixtures whose
answer is known. Its inputs are written rather than collected — `traces.jsonl`
records the *source* of a collection and never its content — which is stated at
the top of the file, because a harness that produced a confident verdict from its
own boilerplate is a thing that has already happened here twice.

## Aggregation by subject (v3.19)

The hot tier carries every deliberate entry, so a completeness question no longer
depends on retrieval. What five real runs on 2026-08-26 also established is that
carrying them is not enough when they overlap: `Quels sont tous les ordinateurs
que je possède ?` came back three machines out of three from eleven lines, and
`Tu peux me lister mon matériel ?` came back with one entry, then two, then one,
against these three:

    - [fact] Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM
    - [fact] NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible
    - [fact] Le NiPoGi a 32 Go de RAM

That is a judgement of SCOPE, and a defensible one — no pair is identical, and
there are zero exact duplicates in the whole store, so `remember_many`'s
duplicate check (the only merging this codebase had) finds nothing. GBNF was the
candidate structural fix and the computers run closed it: the model enumerates
correctly when the lines do not overlap.

**Read the rest of this section with one thing in front:** the tier folds the
block, it does not answer that question, and on that question it currently costs
something. Measured on 2026-09-11, before and after a real fold, `Tu peux me
lister mon matériel ?` went from naming two machines of three to naming one — the
answer stays inside the aggregated entry. What the fold buys is eleven
overlapping lines becoming eight, and no model call.

### What happens to the sources, which had to be settled before any code

Replacing them is destructive and irreversible on entries a human typed. Leaving
them alongside grows the block and restores the overlap. Neither, therefore: a
source is **linked** to the entry that now speaks for it (`superseded_by`), and
exactly one reader — `rag.hot_entries` — skips it.

Both retrieval channels keep superseded rows in scope, and that is the property
the design rests on rather than an oversight. A fold can lose a nuance no
arithmetic sees; if it also dropped the source out of retrieval, the store would
answer *je n'ai rien* while the text sits in it. Measured on a copy after the
first real fold: `#17`, `#307` and `#315` all still come back, on both channels,
for the questions they answer. Word containment
does not imply vector reach either — this page measured the opposite when a fact
was given the word `matériel` and came back at rank 109. **The block gets
shorter; nothing gets harder to find.**

A link and not `status='superseded'`: the flag cannot say by what, so nothing
can audit a fold and undoing one is guesswork. `!memory` shows `#312 [fact] ->
#341`; `!forget 341` releases everything #341 spoke for.

### Two halves, and both of them turned out to be arithmetic

Choosing which entries belong to one subject is enumerable, so it is arithmetic:
the largest set of entries sharing one informative word is a subject, take it,
remove those entries, repeat. Rarest-first was the obvious reading and it splits
the group it aims at — the rarest token shared by two NiPoGi entries is `06`,
out of AM06PRO, which names two of the three and leaves the third alone forever.

Writing the line was the other half, and it was a 9B under a GBNF grammar until
2026-09-11. Two runs of the harness against a copy of the real store settled it:
**seven model calls, zero aggregates written**, and the failure was not a lottery
but the same shape every time — the answer ran to the grammar's maximum item
count. `nipogi` came back as its three notes concatenated, `steam` as a cycle of
three items repeated four times, and once as `Steam Deck` glued to the front of
every item until the cap stopped it.

    Le NiPoGi a : 32 Go de RAM, Matériel NiPoGi AM06PRO, processeur Ryzen
    5500U, 32 Go de RAM, SSD 256 Go, NiPoGi AM06PRO, Arch, 5500U, [...]

    Steam Deck SteamOS : Steam Deck, SteamOS, Podman, Steam Deck, SteamOS,
    Podman, Steam Deck, SteamOS, Podman, Steam Deck, SteamOS, Podman.

Nothing in a lexicon of **words** makes reusing a word expensive. "Say each thing
ONCE" was therefore a rule of the prompt, enforced by nothing, set against "do
not leave anything out", which was enforced by a gate. A model does what the
enforced half asks.

So the grammar's unit moved up one level, to the **detail** — each candidate
detail emittable at most once, in order, the list ending when they run out.
Repetition and the runaway stopped being samplable and the calls dropped from
18–33 s to 6–7 s. One failure survived: on the NiPoGi group the model **omitted**
`32 Go de RAM` and `Arch`, and labelled the list `NiPoGi` rather than `Matériel`,
losing the one word this page records as the reason `#307` is reachable at all on
the word channel. On the `steam` group it produced, to the character, what the
arithmetic produces for free.

That is the finding, and it is the same one this page has recorded about
grammars, thresholds and rewrites: **once the choice is enumerated, the only
freedom left to the model is to omit.** The line is composed in `aggregate.merge`
now, out of the details the sources already contain, and the pass costs nothing.

### A detail is only ever dropped by a detail that says more

`labelled` reads a note as an optional label and its comma-separated details —
the shape the store is already written in. `distinct` deduplicates them under one
rule, and it is the safety property the whole tier rests on: **a detail is
dropped only in favour of a detail that contains every one of its words,
connectives included.**

Informative-word containment was the obvious reading and it is wrong in a way no
arithmetic recovers from. `pas`, `ne`, `jamais`, `sans` are connectives by
frequency — exactly the words an informative-word rule ignores — so under it
`Le NiPoGi n'a pas 32 Go de RAM` is contained in `Le NiPoGi a 32 Go de RAM` and
gets folded into its own opposite, in a store that would then answer the
contrary of what it holds. Requiring every word inverts that: a negation carries
a word its positive does not, so the negation can never be the one dropped.
Nothing that says more is ever deleted by something that says less — with no
negation list, no stopword list and no French, which matters for the same corpus
of NiPoGi, busctl and aardvark-dns that the frequency count exists for.

Two details whose word sets are equal are the one case containment cannot order
(`SSD 256 Go` against `SSD 256Go`, which tokenize identically here). The
spelled-out surface wins, on the one axis this page has measured: an
instruction-tuned embedding model retrieves a telegram worst, and `#313` spent
six campaigns unreachable for being one.

### Sometimes there is nothing to write at all

The real store's `steam` subject is two entries and one of them is the other's
opening words:

    - [fact] Possède un Steam Deck
    - [fact] Possède un Steam Deck sous SteamOS, fait tourner des conteneurs Podman dessus

The merge writes a correct line for it and the budget gate refuses it, correctly:
the line was `#317` with a head glued on, 27 estimated tokens against 33. So when
every surviving detail belongs to one entry, that entry **speaks for the others
as it stands** — nothing composed, nothing stored, one line fewer in the block
and a sentence the user typed left in it.

Quorum and budget do not apply there, because both exist to judge new text.
Quorum refuses an entry standing in for a single other entry, on the grounds that
it is a rewrite of somebody's note; absorbing rewrites nothing, so one source is
enough. The one thing absorption still checks is the namespace: an entry folded
into an entry of another project drops out of the block under a name nobody filed
it with.

### The gates, and what each failure looks like

| gate | what it checks | what happens |
| --- | --- | --- |
| closure | every word of the aggregate is in its lexicon | the subject is abandoned — **cannot fire** under a writer that only copies |
| repetition | no pair of informative words is used twice | the subject is abandoned |
| coverage | every informative word of a source is in the aggregate | that source stays active and the rest still fold — **cannot fire** |
| quorum | at least `COMPACTION_AGGREGATE_MIN_SOURCES` sources fold | nothing written — **cannot fire**, since nothing is held back |
| budget | the aggregate is shorter than what it folds, by more than the token estimator's error | nothing written |

Three of the five cannot fail any more, and they stay where a gate on a model
used to be. They are the assertions that say so: the day anything here composes
a word rather than copying one, the write is refused rather than discovered
later in the store. The retry that used to sit behind two of them is gone with
the call it was retrying.

Repetition is the one the arithmetic can still fail on its own output, and what
it catches has changed. It used to catch a model concatenating its notes; it now
catches two details saying one thing in words that share no set — `32 Go de RAM`
against `32 Go de mémoire` — which set arithmetic cannot merge and must not
pretend to have. Refusing leaves the block with the two overlapping entries it
already had, which is the outcome this tier is meant to improve on and not the
one it is allowed to fake.

The token estimator drifted 21.4%, 21.5% and 16.2% across the model runs, which
is why the budget gate takes a margin rather than a `>=`: the first version
refused nothing and folded a two-entry group for a saving of three estimated
tokens.

### What no gate can see, which is why they are not a truth check

Under the grammar, this sentence passed closure, coverage, quorum **and** budget,
and folded three entries a human had typed:

    Le NiPoGi AM06PRO, un matériel de la NiPoGi AM06PRO, est un processeur
    Ryzen 5500U, 32 Go de RAM, SSD 256 Go, Arch, Ansible, services Podman.

A mini PC is not a processor. **No arithmetic on words will ever see that**, and
none of these gates is a truth check. What the run changed was the shape the
model was asked for: a sentence needs a verb, a verb makes a copula reachable,
and a copula makes a false copula reachable. A labelled list — `head : item,
item` — has no verb slot at all, and it is what the store already holds.

The class of error did not disappear when the model did. Every word of a line is
now one the user typed, which is a strong property and not that one: a true
detail and another true detail can still be put side by side into a sentence
nobody meant. Read what the harness prints.

### The block getting shorter is not the point

The first fold measured end to end made the answers **worse**. The block went
from 11 entries and 195 tokens to 8 and 166, and the hardware question stopped
mentioning the RAM.

The line was `Matériel : Le NiPoGi a 32 Go de RAM, NiPoGi AM06PRO, processeur
Ryzen 5500U, ...` — and what the model dropped is its own first item. `Matériel`
and the list under it are one line typed by one person; another entry's sentence
between them reads as part of the label rather than as an item. So the details of
the entry that gave the head come first, and the others follow.

Measured 2026-09-11 on the block the real store produces, three passes with the
arms rotated between them — llama-server keeps one slot, so a prompt repeated
back to back measures the KV cache and not the model. Each cell is how many of
the three passes got it:

| block | `matériel` keeps 32 Go de RAM | `matériel` names machines | `tous les ordinateurs` names three |
|---|---|---|---|
| not folded | **3/3** | two of three, 3/3 | **3/3** |
| folded, details in source order | **0/3** | two of three, 3/3 | 2/3 |
| folded, the head's own list first | **3/3** | **one** of three, 3/3 | **3/3** |
| folded, informative-word dedup | **3/3** | one of three, 3/3 | 0/3 |

Three things to take from that table, and the last one matters most. The ordering
rule is worth its line of code. The fourth row is why `distinct` compares every
word: dropping `Le NiPoGi a 32 Go de RAM` in favour of `32 Go de RAM` is what
informative-word containment does, it is what folds a negation into its opposite,
and it lost the Steam Deck on the computers question three times out of three —
the safe rule and the one that measures better are the same rule.

And **nothing here beats not folding.** The shipped order ties on the computers
question and loses a machine on the hardware one: an answer that used to name the
NiPoGi and the Dell now names the NiPoGi alone, because the aggregated entry is
where the model stays. The scope judgement this page has recorded since v3.18 is
therefore not solved by this tier — it is, on one question of two, made slightly
narrower. The tier's case is the block it keeps small as the store grows, and it
should be turned on for that reason or not at all.

### Where it runs, and what it costs

In compaction, after the strategy has committed, only when a compaction actually
happened. Not on the write path, because the router normalises what it writes
(measured 2026-08-25, byte-identical output with the category word gone). Not on
the recall path, because the hot block is a stable prefix whose prefill is paid
once — ~180-192 tokens of a 195-token block survived in the KV cache across three
consecutive runs — and a pass that rewrote it mid-conversation would cost that
every turn. Losing the model call removed the latency argument and left that one
standing: a block that changes between two recalls is re-prefilled whatever wrote
it.

No model call, no provider to be down, and the pass still swallows its own
failures: a compaction that has already committed must not become an error the
user reads.

### Earning the knob

`COMPACTION_AGGREGATE` ships false, like every mechanism on this path before it.
Not for what it costs any more, but for what it writes: an entry in this store is
read as something the user said, and it stays.

    podman cp data/forge_rag.db forge:/tmp/real_copy.db
    bench/in_container.sh rag_aggregate --db /tmp/real_copy.db
    bench/in_container.sh rag_aggregate --db /tmp/real_copy.db --apply

Without `--apply` the harness writes nothing and now shows everything: the
groups, the exact line each subject would fold into, the gate that would refuse
it, and what the block would weigh either way. It runs the real pass with the
write turned off rather than a second copy of the gate sequence in a harness,
because a harness that computes the fold separately is one that can be right
about a pass that is wrong. With `--apply` it writes **to the copy**.

Measured on a copy of the real store, 2026-09-11:

| | |
|---|---|
| subjects | 2 |
| written | 1 (`nipogi`, 3 sources, 57 → 36 estimated tokens) |
| absorbed | 1 (`steam`, `#1` into `#317`, nothing written) |
| refused | 0 |
| block | 11 entries / 195 tokens → 8 / 166 |

And the sources are still reachable after the fold — `#17`, `#307` and `#315`
all come back on both channels for `Quel processeur a mon NiPoGi ?` and `Combien
de RAM a le NiPoGi ?`, superseded and all. The block gets shorter; nothing gets
harder to find.

### Trying it for real

The same discipline the measurement campaigns needed, now for live trials:
`!clear` **before** the history crosses the compaction threshold. Otherwise the
trial question is still in the window when compaction fires, its own exchange
gets archived, and the next trial's best match is the previous trial. Four
transcripts of `Tu peux me lister mon matériel ?` reached the store that way at
0.5757–0.7312 — three refusals and one answer — on messages persisted on
2026-08-22, the day before `forge/outcome.py` existed to mark them.

That gap is the general point and it outlives this branch: the mark is applied at
persist time and read at compaction time, and those two moments can be days
apart. **Any rule about what compaction may index has a latency equal to the
lifetime of the rolling history.** A filter merged today protects nothing already
sitting unmarked in `memory.json`.

`!clear` also empties the tiroir, which is a known debt and not this branch's.

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
