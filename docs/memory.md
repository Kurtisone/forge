# Memory, RAG and traces

Forge has three distinct stores and they are easy to confuse: the
conversation history (one global file), the vector store (facts and
compaction pointers), and the execution traces (one line per run).

What each mechanism here was measured against, and the campaigns that ended
with a knob shipping **off**, are in [The measurement record](campaigns.md).
This page says what is true now; that one says why, and what has already been
tried.

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
entry, whose one vector was the *average* of a dozen chunk vectors — which is close
to no question in particular. `forge/transcript.py` cuts the window at user turns (a
user turn plus whatever answered it is the smallest unit that still stands on its
own) and `rag.remember_many` stores the pieces as rows: the same number of embedding
requests, kept apart instead of averaged. Two things never reach the store — an
earlier compaction pointer, which answers no question, and a raw router-JSON
envelope, unwrapped to its content — and both the live eviction and the migration go
through the same `transcript.units` / `transcript.split` pipeline, so they cannot
filter differently. `bench/rag_dilution.py` measures the difference and
`deploy/rag_resplit.py` re-slices what was written before the change; the numbers
that earned it, and what they did to `RECALL_MAX_DISTANCE`, are in [The measurement
record](campaigns.md).

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

**A turn that is a question never writes, whatever the payload says.** That rule is
in code and not in the prompt, because the prompt already had it. From
`traces.jsonl`, 2026-09-11:

    user: Je possède un serveur ?
    payload: {"action":"remember","kind":"fact","content":"Possède un serveur"}

The question mark is the only thing separating that turn from a statement, and the
payload does not carry it — so nothing downstream of the router can tell what
happened, and the entry sits in the store as something the user said about
themselves. Ask it again a week later and it is the store's best match for its own
question, which is the failure this page records from three other directions.

So the test is on the TURN, which is the one thing the payload cannot lose
(`forge/turn.py`, the same module delegation reads for the same reason), and the
question is answered with a recall instead — that is what it was asking for.
Measured over every memory routing in the trace file before shipping it: 19
remembers, 18 of them declarative and legitimate, one a question, and it is the
bug. No genuine remember has ever been phrased as a question here, so the rule
costs nothing that was measured to exist. A long message merely *ending* in a
question mark still writes, on the same bound delegation has always used.

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

Both were narrower than they read until 2026-09-12, and what widening them
recognised — 13 entries of 195 — is in [The measurement
record](campaigns.md#both-filters-were-narrower-than-they-read-measured-2026-09-12),
along with the residue that is still open: nine refusals the model wrote in its
own words, five of them a chat turn declining, which no phrase list can reach
without starting to drop real answers.

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

What this channel reached, what it dragged in, and why it was later found to
return nothing the hot block does not already hold: [The measurement
record](campaigns.md#two-channels-two-admission-rules-v317).

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

## The hot tier (v3.18)

Six campaigns end on one sentence, and [the record](campaigns.md) states it from
several sides: **retrieval answers proximity questions, not
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

It is the one retrieval knob the measurements point at: on 2026-09-12 the block
was found to return every entry the word channel does and every entry the
expansion pass was built to rescue, for a prefill paid once. The numbers, and
the two cases where that stops being true, are in [The measurement
record](campaigns.md#the-hot-tier-and-what-it-subsumes-v318-concluded-2026-09-12).

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

The writer is arithmetic and not a model, which is a correction the tier took
after shipping: seven model calls on the real store wrote zero aggregates. That
campaign, and the end-to-end measurement that keeps `COMPACTION_AGGREGATE` off,
are in [The measurement
record](campaigns.md#aggregation-and-the-writer-that-stopped-being-a-model-v319).

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

## Reading and repairing the store

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
`total_ms`, `ok`, `not_answered`, `error`, and `llm` (per-run inference totals, including
which model actually answered).

`ok` and `not_answered` are not the same question, and reading the first as the second is
a bug this file exists to prevent a repeat of. **`ok` is a rendering directive**: it says
the run should be surfaced as a message rather than as a crash, and graphs set it true on
purpose when they fail — `graphs/recall.py`'s error node carries the comment. So a run can
be `ok` and have answered nothing at all.

The `llm` block carries `models`: what actually answered, as each backend names it.
The web UI shows it on the trace card, which is a different question from the one the
header answers — the header names the model loaded **now**, and a trace can be from
before a swap.

**`not_answered`** is the verdict: the reason the run produced no answer, or `null` if it
did. It is the *same* value that keeps the exchange out of the vector store, computed once
on the single exit path — once being structural rather than tidy, since `outcome.taken()`
clears on read and a second caller would get `null` and disagree with the first.

Observed before it existed: the router invented a hostname, `web_fetch` returned
`[error] could not resolve host` as its output string, the log recorded
`memory.not_indexed reason='non-answer reply'` — and the same run was drawn with a green
tick. The web UI now has three states for this: ✓ answered, ∅ ran and answered nothing,
✗ failed.

---

[← Documentation index](README.md) · [← Project README](../README.md)

