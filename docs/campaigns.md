# The measurement record

Every retrieval mechanism in Forge ships **off** until a harness in
[`bench/`](../bench/) has measured it against a copy of the real store, and
several ship off *because* of what the harness said. This page is that record:
what was tried, what the numbers were, and what each campaign settled.

It is deliberately separate from [Memory, RAG and traces](memory.md), which
says what the three stores do and how to work with them **today**. That page
answers "what is true now"; this one answers "why", and "what has already been
tried, so nobody runs it again".

Read a section's first line: it says whether you are reading a live rule or a
recorded dead end. A dead end is kept at the length that makes it
reproducible — that is the whole point of writing it down.

The harnesses themselves, and how to run one, are in
[Development](development.md#the-measurement-harnesses).

## Per-exchange intake (v3.14)

The first campaign, and the one that made the others measurable: until v3.14
compaction stored an evicted window as a single entry, whose one vector was the
average of a dozen chunk vectors.

### One entry per exchange, not one per block

**Shipped on, and it is the intake every later campaign measured against.**

Measured on the real store, six questions before and after re-slicing: hits moved
from 0.8934 / 0.671 / 0.9386 to 0.4498 / 0.671 / 0.7695, turning a **NO GAP**
verdict (worst hit closer than the best miss) into a usable gap of 0.0642. The
short fact at 0.671 did not move — only the entries that had been buried did, which
is the control. That is what makes `RECALL_MAX_DISTANCE` a number one can choose;
see `.env.example` for the value and why it is not the harness's midpoint.

An earlier measurement, from the same store on 2026-08-22, is what opened it: a
question sat at 0.9386 from the compacted block containing it nearly word for word,
while a short fact sat at 0.671. The block was not a worse entry than the fact — it
was a dozen entries averaged into one, and the average answers no question in
particular.

`bench/rag_dilution.py` reproduces the comparison, and `deploy/rag_resplit.py` is
the one-shot migration for blocks written before the change. How the intake works
today is in [Memory, RAG and traces](memory.md#context-compaction--drawer-v39).

## The threshold, and what no threshold answers (v3.15-v3.16)

Six rounds of query expansion, and the finding that closed them: on a
store of a few short overlapping facts, every distance bunches and no
rewriting separates an answer from a neighbour.

### When the question is the problem, not the threshold

**Opened the expansion campaign. The finding stands; the mechanism it led to does not** — see below.

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

**Verdict: `RECALL_EXPANSION` ships `off`.** Re-measured 2026-09-12 against the hot block: it now returns nothing at all.

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

**The fix that made the mechanism correct, on a mechanism that is still off.** The grammar rule itself is live and general.

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

The entries no phrasing reaches, and what searching the words costs.

### Measured on the real store, 2026-08-25

**Won on its own terms, superseded 2026-09-12** by the hot block (SUBSUMED 7 / ADDS 0).

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

**The limit that opened the hot tier.**

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

**Why `RECALL_LEXICAL` shipped `off` in the first place.**

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

## The hot tier, and what it subsumes (v3.18, concluded 2026-09-12)

Carrying the whole deliberate store instead of searching it, and the
question that took three weeks to answer: what is left for the two
rescue mechanisms to do.

### What five real runs established

**Validated the hot tier and opened the aggregation tier.**

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

**A discipline for live trials, still current** — the short version is in [Trying it for real](memory.md#trying-it-for-real).

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

**The measurement that became the 2026-09-12 conclusion below.**

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

### Measured together, 2026-09-12, and now concluded

**The conclusion. `RECALL_HOT_FACTS` is the one to turn on; the other two pay twice for it.**

`bench/rag_hot_tier.py --cutoff` runs both rescues against the same
block and reports them in the same two columns. On the real store as it
stands (193 entries, 11 deliberate), with `#307` and `#317` named as
the answers:

| | fired on | cost | SUBSUMED | ADDS |
|---|---|---|---|---|
| word channel | every question | one FTS5 query | **7** | **0** |
| expansion | 3 questions of 4 | **28.9 s** | 0 | 0 |

**The word channel returns nothing the block does not already hold**,
and on this store that is close to a tautology rather than a
coincidence: `RECALL_LEXICAL_EXCLUDE_ARCHIVED` ships `true`, so it only
ever returns non-archived rows, and the block is exactly the set of
non-archived rows. The measurement is worth having anyway, because the
identity breaks in two cases the harness reports directly — when the
block **truncates** (the cap stops being a tripwire) and when an
aggregate **supersedes** a row, which the block skips and the word
channel still reaches.

**The expansion pass returned nothing at all**, which is a different
finding from subsumed and the harness now says so. It applies the same
cutoff to its own results, and the nearest distances on the three
questions where it fired were `0.8899`, `0.9581` and `0.9979` against a
cutoff of `0.88` — the v3.16 result reproduced on a store that has
changed since. Twenty-nine seconds, three model calls, zero rows.

And the column that settles it: **every `--expect` entry was in the
block.** Both mechanisms exist to reach `#307` and `#317`; the block
carries them unconditionally, for a prefill paid once.

So, as defaults on this store: `RECALL_HOT_FACTS` is the one worth
turning on, and it is the only one of the three that delivered anything
here. `RECALL_LEXICAL` stays off while the block is on, and becomes
worth re-measuring the day the block truncates or the aggregation tier
starts superseding rows. `RECALL_EXPANSION` stays off, and its own
precondition already made it inert in production — with no
`RECALL_MAX_DISTANCE` set, nothing is ever dropped, so there is never a
failure to rescue.

None of that is an argument for deleting either mechanism. They were
both measured and both won on their own terms against the store of
their day; what changed is that a later tier answers the same questions
for less. The harness is one command, and the day the block stops
covering the store it is the thing that says so.

There is deliberately **no suggested budget** in that output. The cap
is a budget and not a measurement; the only thing that moves it is the
store growing, and `HEADROOM` reports that directly.

## Aggregation, and the writer that stopped being a model (v3.19)

Seven model calls, zero aggregates, and what replaced them.

### Two halves, and both of them turned out to be arithmetic

**Why the aggregation writer is code and not a model.** The shipped design is in [Aggregation by subject](memory.md#aggregation-by-subject-v319).

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

### What no gate can see, which is why they are not a truth check

**Still true of the shipped writer: no gate here is a truth check.**

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

**Why `COMPACTION_AGGREGATE` ships `off`.**

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

## What running it in anger found (v3.20)

Five faults, all the same shape: a rule the model was asked to follow,
replaced by one it cannot break.

### Both filters were narrower than they read, measured 2026-09-12

**Both filters were widened. The residue is named at the end and is still open.**

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

**Verdict: no verdict channel at this model size.** One deterministic case came out of it and shipped.

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

---

[← Documentation index](README.md) · [← Memory, RAG and traces](memory.md)
