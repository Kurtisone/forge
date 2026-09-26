# Version history and roadmap

The product roadmap. It is deliberately separate from
[ARCHITECTURE.md](../ARCHITECTURE.md): the architectural maturity axis
(Kernel, capabilities, scheduler) advances independently of releases and
conflating the two produced a version number that meant two things.

The table is an index and nothing more. It stopped being one around
v3.11, when rows started carrying the reasoning behind a release as well
as its contents -- v3.20 reached 4346 characters in a single cell, which
is prose wearing a table's clothes and unreadable as either. What each
release actually did is below, one section per version, and the
measurements behind those decisions are in
[campaigns.md](campaigns.md), where every number in this file already
lived.

## Roadmap

| Version | Status | Focus |
|---|---|---|
| **v2.2** | done | Clean Runtime: typed errors, centralized logger, provider split, loop guard |
| **v2.3** | done | Robustness: parser cascade, memory hardening, REPL paste detection, ENABLED_TOOLS allowlist |
| **v2.4** | done | Structured execution trace: `AgentState`, `TraceStep`, JSONL trace file, `!trace` |
| **v3.0** | done | Graph execution engine: `Node/Edge/Graph`, conditional edges, `AgentState.context` |
| **v3.1** | done | HTTP API + web UI, review graph, `forge review` CLI, sandboxed files tool |
| **v3.2** | done | Shell tool, git tool, `POST /run`, Tools tab in UI |
| **v3.3** | done | Hardening: real multi-step orchestrator, CI (ruff + pytest), optional API bearer-token auth |
| **v3.4** | done | Portfolio: architecture diagram, `.env.example`, LinkedIn writeup |
| **v3.5** | done | Test coverage (llm/cli/trace: 26-39% → 98-100%), router reachable to files/shell/git, API rate limiting |
| **v3.6** | done | Response quality: GBNF grammar-constrained decoding for llama.cpp |
| **v3.7** | done | Vector memory / RAG: SQLite-vec, `/remember` + `/search`, `!remember`/`!recall` REPL commands, a router-dispatchable `memory` tool, Qwen3-Embedding-0.6B |
| **v3.8** | done | Prompt-cache reliability: pinned slot, and a sliding window that was fighting it — [detail](#v38--prompt-cache-reliability) |
| **v3.9** | done | Context compaction + drawer: `rag_pointer`/`llm_summary` strategies, pin/unpin, `/history` `/drawer` `/compact` endpoints, `!compact` REPL command, files write-diff |
| **v3.10** | done | Hardening plus four tools: `test`, `web_fetch`, `web_search`, `research` — [detail](#v310--hardening-and-four-new-tools) |
| **v3.11** | done | Sysadmin: read-only diagnosis from real logs, and three proxies to reach them — [detail](#v311--sysadmin) |
| **v3.12** | done | Instrumentation, and a router prompt made a pure append (~3.9x on warm turns) — [detail](#v312--instrumentation-and-router-latency) |
| **v3.13** | done | Delegation to Claude Code: a drafted spec and a job with a checked lifecycle — [detail](#v313--delegation-to-claude-code) |
| **fix/dettes-v3.12** | done | Six fixes found by running the previous two versions in anger — [detail](#fixdettes-v312--six-fixes-found-in-anger) |
| **v3.14** | done | Per-exchange intake: one entry per exchange instead of one averaged block — [detail](#v314--per-exchange-intake) |
| **v3.15** | done | What the store may hold, and the number that judges it — [detail](#v315--what-the-store-may-hold) |
| **v3.16** | done | Query expansion, measured and shipped **off** — [detail](#v316--query-expansion) |
| **v3.17** | done | A word channel beside the vector one, admitted on the query — [detail](#v317--two-channels-two-admission-rules) |
| **v3.18** | done | Memory hot tier: the deliberate store goes in whole, unsearched — [detail](#v318--memory-hot-tier) |
| **v3.19** | done | Aggregation by subject, composed in code rather than written by a model — [detail](#v319--aggregation-by-subject) |
| **v3.20** | done | Five faults found in anger, plus 3.20.1 and 3.20.2 — [detail](#v320--five-faults-found-in-anger) |
| **v3.21** | done | Forge stops assuming which model it is talking to, and the trace stops assuming a run answered — [detail](#v321--forge-stops-assuming-which-model-answered) |
| **v3.22** | done | One backend per capability, and three UI bugs a human found in ten minutes — [detail](#v322--one-backend-per-capability-and-what-a-human-pass-is-for) |
| **v3.23** | done | Phone pairing, and the Harnais: `sysadmin` reads observed facts instead of raw text, and says when it could not observe — [detail](#v323--the-harnais-and-the-first-time-production-measured-it) |
| **v3.24** | done | A fourth collector, from a reply Forge was already throwing away — [detail](#v324--a-unit-domain) |
| **Kernel L2** | done | Capability layer and a deterministic Policy Engine, both wired into the orchestrator, the API, the CLI and the graphs — see [ARCHITECTURE.md](../ARCHITECTURE.md) and [The Kernel layer](architecture.md#the-kernel-layer). Sits on the architectural maturity axis, not this product roadmap |
| **Kernel L3** | blocked | The Cognitive Scheduler, and the reason it is not started: every capability resolves to exactly one candidate, so there is nothing to arbitrate. `_dispatch` says so in code, and stops hard rather than picking silently. `CAPABILITY_PROVIDER` (v3.22) moves the choice from the process to the work without inventing the arbiter |

---

## What each version did

Only the releases whose story does not fit on one line. The
others are complete in the table above.

### v3.8 — prompt-cache reliability

Prompt-cache reliability: pinned llama-server slot, `MEMORY_MAX_HISTORY`
raised to stop a sliding window from fighting KV-cache reuse —
root-caused a remaining cache-reuse gap to the served model's own hybrid
architecture, not Forge

### v3.10 — hardening and four new tools

Hardening + new tools: dedicated `test` tool, `web_fetch`
(SSRF-guarded), `web_search` + `research` (self-hosted SearXNG), review
graph gains an optional test-run step and chat-dispatch; router
disambiguation fixes (files vs review, tool descriptions/examples for
every new tool) found through real usage

### v3.11 — sysadmin

Sysadmin: `discover → collect → synthesize` graph diagnosing real
service/system problems from logs, read-only always (no restart/stop
path exists in the code); UI gains expandable per-step detail
(`forge.subtrace`) for every graph-based tool; read-only host access via
three independent proxies (`xdg-dbus-proxy` for systemd, a hand-rolled
GET-only proxy for podman.sock, a plain bind mount for the journal)
rather than raw socket access — real production debugging found and
fixed a prompt-injection-shaped example-leak, a context-overflow crash,
`systemctl`'s undocumented refusal to honor `DBUS_SYSTEM_BUS_ADDRESS`
(switched discovery to `busctl`), and a rootless-podman
supplementary-groups gap blocking `journalctl -u` on root-owned services
(`--group-add keep-groups`)

### v3.12 — instrumentation and router latency

Instrumentation + router prompt latency, five batches: per-call token
accounting from llama-server's own timings; the router prompt made a
**pure append** over the previous turn, taking warm prompt processing
from 3.12 to 0.81 ms/token (~3.9x, ~8.4s to ~2.2s per routing call) with
zero regressions across a 29-fixture A/B (`bench/router_ab.py`); a
permanent context gauge in the header with `GET /context`; compaction
triggered on a token budget rather than a message count; and file-path
grounding in the orchestrator, refusing a write or a review on a path
the model invented

### v3.13 — delegation to Claude Code

Delegation to Claude Code: a spec drafted under its own GBNF grammar,
persisted jobs with a checked lifecycle (`draft → awaiting_user → ready
→ running → done`), the model asking follow-up questions rather than
inventing missing detail, cancellation, and a runner thread that hands
off without blocking the conversation. Entry is a `delegate` tool, so it
stays reachable from the chat thread like everything else

### fix/dettes-v3.12 — six fixes found in anger

Six fixes found by running the previous two versions in anger:
`test_path` was invisible to the path guard (and so was a raw `pytest
<path>` command string), a pasted paragraph in `file_path` is text
rather than an unfounded path, `load_memory()` died on valid JSON of the
wrong shape, `recall` answered in the wrong language (new
`forge/lang.py`, closed-vocabulary fr/en detection that stays silent
when unsure), and the `test` tool had neither a description nor an
example in the router prompt

### v3.14 — per-exchange intake

Per-exchange intake: compaction stored an evicted window as ONE entry,
so its single vector was the average of a dozen chunks and matched no
question in particular. `forge/transcript.py` cuts at user turns and
`rag.remember_many` stores the pieces as rows -- same number of
embedding requests, kept apart instead of averaged. One pipeline for the
live path and for `deploy/rag_resplit.py`, which re-slices what was
already written, because two filters that must agree are how a fix
reaches one of them. Measured on the real store, six questions: hits
moved from 0.8934 / 0.671 / 0.9386 to 0.4498 / 0.671 / 0.7695, turning a
**NO GAP** verdict into a usable 0.0642 -- and the short fact that did
not move is the control

### v3.15 — what the store may hold

What the store is allowed to hold, and the number that judges it. The
query instruction goes on the QUERY and never on the document
(Qwen3-Embedding is asymmetric), which needed no migration for exactly
that reason. A recall answer is never indexed: it was rebuilt from
entries the store already holds, and the copy carries the *question*,
which outranks the answer. Two filters of different kinds keep refusals
out -- `forge/outcome.py` (the run says so, survives rewording) and
`forge/non_answer.py` (the text says so, the only test `rag_resplit`
has).

**3.15.1**: `RECALL_MAX_DISTANCE` carries the embedding regime it was
measured in (`0.88@a5c47b`) and turns the cutoff OFF rather than
adjusting it when the tag is stale -- a number from another regime is
not too high or too low, it is unrelated. Five harness faults found in
one session are why `bench/_harness.py` exists

### v3.16 — query expansion

Query expansion, measured and shipped **off**. `terms` (deterministic
rewrites) lost on four questions out of four: the embedding model is
instruction-tuned on natural language and a keyword bag is
off-distribution. `llm` works as a mechanism -- `#307` went from absent
to rank 1 -- and cannot *separate* on a store of six short overlapping
facts, where every distance bunches between 0.94 and 1.14. Six rounds,
three readings of the gap, negative every time.

**3.16.1**: the rewrites are complete questions and the GRAMMAR enforces
it, not the prompt; the write path stopped storing the telegraphic shape
the read path finds worst. The order of work is fill the store, then
calibrate the rescue

### v3.17 — two channels, two admission rules

Two channels, two admission rules: an FTS5 index over the same rows,
unioned on entry id, for the entries no phrasing reaches. The word
channel admits on the **query** (`RECALL_LEXICAL_MAX_DF`, a share of the
store) rather than on a bm25 score, because a score cutoff would be a
second calibrated number needing its own tag and its own re-measurement
every time the store grows. No fused score, and the distance cutoff
applies to vector rows and to nothing else -- judging a word match by a
distance would delete precisely what the channel was added to reach.
Measured 2026-08-25: `#307` and `#313` reached for the first time,
`MAX_DF` changed nothing between 0.05 and 0.5, and everything the
channel dragged in was archived transcript quoting the question back

### v3.18 — memory hot tier

Memory hot tier: every deliberately-written entry goes into the recall
synthesis prompt whole, above the question, unsearched — a completeness
answer where six campaigns of proximity tuning could not give one.
Validated in the sharpest form available: on `Quels sont tous les
ordinateurs que je possède ?` retrieval returned **nothing** (0.959,
0.967, 1.023, `results=0` after expansion) and the answer was complete
and correct, three machines out of eleven lines of which eight are not
computers. The cap is a tripwire, not a policy (11 entries / ~195 tokens
against a 1000 budget), overflow cuts the tail and says so in-band. Two
findings the branch did not expect: the prefill is paid **once** and not
per recall (~180-192 tokens of a 195-token block surviving in the KV
cache across three real runs, a router call in between failing to evict
it), and the ranking removed from retrieval **reappears in the model** —
`liste mon matériel` answers from one entry of eleven, which the
computers run identifies as a scope judgement rather than an enumeration
failure, closing GBNF as a fix and handing it to the aggregation tier.
`RECALL_HOT_FACTS` ships off; `bench/rag_hot_tier.py` earns it and
reports `SUBSUMED 4 / ADDS 0` against the word channel — redundancy
measured, decision deliberately not taken

### v3.19 — aggregation by subject

Aggregation by subject: overlapping deliberate entries fold into one
line, and the sources are **linked** to it (`superseded_by`), never
deleted -- exactly one reader, `rag.hot_entries`, skips them, so the
block gets shorter and nothing gets harder to find. Grouping is
arithmetic (the largest set sharing one informative word, which must
share more than its name). Writing the line was a 9B under a GBNF
grammar until the real store measured it: **seven calls, zero
aggregates**, every answer running to the grammar's item cap. Moving the
grammar's unit from the word to the detail ended the repetition and left
omission, which is the finding -- once the choice is enumerated, the
only freedom left to a model is to OMIT. The line is composed in code
now, out of the sources' own details, verbatim; a note that says less
than another is absorbed by it with nothing written at all.
`COMPACTION_AGGREGATE` ships **off**, and the reason is measured too:
the fold does not make Forge answer better (`lister mon matériel` named
two machines before, one after), it keeps the block small as the store
grows

### v3.20 — five faults found in anger

Five faults found by running the previous versions in anger, all of them
the same shape -- a rule the model was asked to follow, replaced by one
it cannot break. Both anti-non-answer filters were narrower than they
read: 22 of 195 archived entries were the assistant refusing and
**none** was recognised, because the text filter is a closed set with no
way to discover its own members (four producers registered) and the
structural one was wired into one graph out of five
(`add_node(answers=False)` now carries it). A question the user asked is
never a fact they stated -- `Je possède un serveur ?` had been stored as
`Possède un serveur`, the question mark dropped in passing. A verdict
channel for "these logs do not answer the question" was asked for four
ways and refused: each arm errs entirely on the side its phrasing
invites, so it does not exist at this model size -- what it did produce
is the case needing no judgement, an empty collection. `.device` units
left discovery, where they were four of the eight names the user was
shown. And a web answer now says so when the question named a container
running here.

**3.20.1**: the guard that refuses an invented path was answering with
`ok=True` and `remember=False`, which the web UI cannot render -- it
rebuilds the thread from `/history` after every turn and appends nothing
of its own when the run succeeded, so the user's **question**
disappeared along with the answer that never came. Reported as "no
reply, and it erases my message", which names the symptom and not the
half of it that matters. Persisted now, and kept out of the vector store
by the filter that already knows it rather than by declining to write it
down -- the two are different decisions and they had been made together.
Those same two sentences were also the only fixed refusals Forge wrote
in English, translated here. Two router fixtures (`h01`, `h02`) went in
alongside, for the routing fault that produced the report -- an opinion
asked about a thing, routed to `review` -- which **did not reproduce**:
six replays against a prompt rebuilt to the live 14762 characters, cold
and warm, all six correct. The fixtures stay because a fixture is how an
intermittent fault gets noticed when it becomes systematic, and because
two failed probes on the way there established the rule they are there
to serve: a probe that does not reproduce the prompt does not measure
the routing. What they turned out to measure is narrower and more
useful than either reading of them: `h02` is stable inside a
llama-server process and different across a restart -- `review` all
morning, `chat` twelve times out of twelve the same evening on the same
GGUF, with the prompt verified byte-identical and the cache ruled out
by the harness's own `--no-cache`. Not intermittent, not
deterministic. A tripwire across restarts, whose value is that it
flips.

**3.20.2**: the router's sanity checks were running on the one path a
model does not take. The repetition guard read the whole raw output, the
leak check sat on the plain-text fallback, and a model decoding under
the GBNF grammar returns a valid JSON object at step 1 -- so anything
inside `content` skipped both. The grammar was doing its job;
guaranteeing the SHAPE of the output is what made the output miss the
guards. Found by swapping the served model for LFM2.5-8B-A1B, which
produced nine malformed replies in thirty-six fixtures where the 9B
produced none: a check nothing exercises is not known to work.
Attributing those nine to the model was half wrong, and the correction
is worth more than the original claim -- Forge sends a RAW prompt to
`/completion` and applies no chat template at all, so a model whose
template is ChatML was being run off-distribution from its first token.
Framing the same prompt takes the malformed-envelope count from four to
zero. The bad output was Forge's; the checks it walked through were
still missing. Three faults, the third found by fixing the first two --
the checks now run on a validated object's content, terminal where step
4 could not see the same text and permissive where it can;
`_is_repetition_loop` measures repeated n-grams instead of one token's
share, which was bounded by the repeating unit's own length and so blind
to every loop longer than a word by arithmetic rather than by luck; and
a `chat` content that is itself a router envelope is unwrapped when
complete and refused when truncated. It applies to `chat` alone, because
a tool payload is data and a config file of near-identical lines is a
loop and a good file. Measured against 72 recorded replies: four
decisions change, all four malformed output that used to reach the user,
nothing from the 9B touched. The five left are reasoning leaked into
prose, which nothing text-based catches

### v3.21 — Forge stops assuming which model answered

Four changes that come from one thing happening: the served model was
swapped for the first time, and every place Forge had quietly assumed
there was only ever one came apart at once.

The prompt goes to `/completion` as raw text, so no chat template is
applied and a model whose template is ChatML runs off-distribution from
its first token. `LLAMA_CPP_APPLY_TEMPLATE` asks llama-server's
`/apply-template` once what the loaded model wants and reuses the
answer, so nothing in Forge names a format. It ships **off** and the
reason is the cost, not caution: a constant suffix after the growing
prompt breaks v3.12's pure-append property outright, 11/11 to 0/11, for
a divergent tail of 33 characters -- about eight tokens a turn against
the thousands v3.12 was fighting. Measured on the real flag: malformed
envelopes 3 to 0, routing score 19 to 19, seventeen decisions out of
thirty-one changed. It fixes a shape, not a quality, which is exactly
why it is a knob and not a default.

`LLM_MODEL` is a label that under llama.cpp is never even sent, so a
trace recording it recorded an assumption. Every backend already returns
what actually answered and Forge dropped it at the provider boundary --
the same thing that happened to token counts, and the reason
`types.Completion` exists. `RunMetrics.models` is a list because one run
is several calls and nothing guarantees they were served by the same
thing.

Research names the pages it was built from, listed in code from the URLs
the graph opened rather than asked of the model: a plausible URL a model
produced is indistinguishable from one it read. Pages actually opened
are named; every other search result contributed a snippet and is
counted, never named.

And a run that answered nothing was drawn green. `ok` is a rendering
directive -- "surface as message, not crash" is written at every site
that sets it -- and `_dispatch` returns ok=True for any tool that
returns a string without raising, whatever the string says. Observed
live: the router invented a hostname, `web_fetch` answered `[error]
could not resolve host`, the indexing path recognised it and the trace
drew a tick, three log lines apart. Setting ok False was not the fix
either: `remember` derives from it, and 3.20.1 is the bug report from
the last time those two were confused. The verdict is a separate field
now, computed once on the single exit path -- once being structural,
since `outcome.taken()` clears on read and a second caller would
disagree with the first by construction. The web UI gained the third
state it always needed.

### v3.22 — one backend per capability, and what a human pass is for

`CAPABILITY_PROVIDER` makes the backend a property of the WORK rather
than of the process. `FORGE_PROVIDER` is one value for everything, so a
routing decision, a research synthesis and a compaction summary all go
to the same place whether or not that is a good idea -- and
ARCHITECTURE.md's Niveau 2 is written against a world where they do not.
Empty by default. It is not the Cognitive Scheduler and does not pretend
to be: each capability still resolves to exactly ONE backend, from
configuration, so `candidates()` is untouched and `_dispatch`'s hard
stop on an ambiguous capability keeps meaning what it says. It carries
no cost or quality scores either, for the reason `kernel/capability.py`
already gives at length. A contextvar rather than a parameter, following
`subtrace`, `metrics`, `outcome` and `turn`, which all made the same
trade for the same reason; a context manager rather than set/clear,
because what it must survive is a tool that raises. The grammar warning
moved to the RESOLVED backend, which is the point of moving it.

Then a human clicked around for ten minutes and found three bugs the
suite could not reach. Emphasis written with underscores was arriving on
screen with its punctuation showing, because the UI implements
`**bold**` and `*em*` and nothing else -- the local-container footer had
been wrong since the day it was written. Asterisks rather than a new
rule in the renderer, because this product's answers are full of
`file_path` and `RECALL_MAX_DISTANCE` and emphasis on underscore would
eat identifiers. Switching WINDOW did not refresh the model name while
switching tab did: `visibilitychange` fires on tab visibility, so
alt-tabbing to a terminal to swap a model -- the exact case the listener
was written for -- fired nothing. And a green step pill sat next to
"answered nothing", two accurate facts rendered as one mixed signal.

All three have the same shape: the source assertion was TRUE and the
behaviour was wrong. An assertion can say a branch exists; it cannot say
the branch is bound to the right browser event, or that a second place
is not rendering the same fact differently.

What closed that gap for the renderer is that the "no JS runtime" limit
three test files state was never true of this machine, which has deno.
`tests/test_ui_rendering.py` extracts `formatContent` and its helpers
from `index.html` and executes them -- extracted, not copied, because a
second copy of a renderer is a copy that drifts. It found the underscore
bug on its first run, and it now proves by execution what
`test_ui_security.py` could only assert: that escaping holds and that a
`javascript:` URL never becomes a link.

And `h02` was corrected from both of the confident things said about it.
It is stable inside a llama-server process and different across a
restart -- `review` all morning, `chat` twelve times out of twelve the
same evening on the same GGUF, with the prompt verified byte-identical
and the KV cache ruled out by the harness's own `--no-cache`. Neither
intermittent nor deterministic. A tripwire across restarts, whose value
is that it flips.

### v3.23 — the Harnais, and the first time production measured it

Two lines of work. The second is what this section is about.

The first is pairing a phone. `!pair` renders a QR in the thread that
carries a single-use, short-lived token, and `POST /pair/claim`
exchanges it once for the bearer token. The bearer token is never drawn:
everything printed in the thread is persisted to `memory.json` and
indexed in the vector store, so one rendered there would stay readable,
and retrievable by a `recall`, for good. Around it, for the same client:
`job_id` in `ChatResponse` so an app stops reading the job number out of
prose, a timestamp per turn in history, the model's name in the UI, and
`FORGE_PUBLIC_URL` as a comma-separated list (WireGuard and LAN).

The second is the Harnais. Before it, `sysadmin` handed raw command
output to the model, and the case on record was a downed podman proxy:
`containers=0` read as "no containers", and the diagnosis that followed
was confident and invented. Now collectors feed an in-memory World Model
and a Context Builder writes what the synthesis reads, keeping what was
observed apart from what could not be. Two decisions in it are
structural rather than stylistic. `collect()` returns an `Observation`,
not the `list[Fact]` the design document specifies, because a list has
one way to say "no facts" and a `Fact` cannot carry a failure -- "proxy
down" and "machine idle" would have reached the reader identical, which
is the original bug. And `Fact.confidence` is `field(init=False)`, since
a `Literal` alone enforces nothing at runtime.

It is wired behind `SYSADMIN_USE_HARNAIS` (default `true`) on both paths
of the graph. Route A, no target named, lets the collectors alone feed
the context. Route C, a target named, keeps `_collect_node` as the
validator and records its already-validated output as a `logs`
observation, so no collector gained a target parameter and there is
still no way from router text to `journalctl --unit=`. Both arms were
put to the model in service before each was wired
(`bench/context_builder_ab.py`); the measurements are in
[harnais.md](harnais.md), not here.

Then production measured it. Five real turns read from `/traces`: route
C worked, twice. Route A never ran. Two questions that named no target
came back from the router with `forge-podman-ro-proxy` as `target_hint`,
copied from a shell turn in the history, and route A is gated on there
being no target -- so a phantom one switched it off, and `target_missed`
refused correctly where the Harnais knew the answer. It is the `h02`
habit again: the model copies the prompt rather than guessing. A target
the user never typed now falls back to route A. That fallback has not
been exercised by a real turn; the router got the next question right
unaided, so it is insurance, not field proof.

The rest came from writing the first direct test of functions every
other test replaces with a lambda. `nothing_collected` could never
fire: `_run_fixed` returns the string `[no output]` and the edge tested
for `""`. `running_containers()` threw podman's error text away, and its
first test found an idle machine reporting a container named
`[no output]`. `TOOL_ERROR_PREFIX` was the one refusal marker in the
`non_answer` registry whose removal broke no test (14 of 15 did).
`turn.clear()` was promised by `turn.py` and called by nothing, so
`/run` read the message of the last `/chat` that landed on its worker.
The subprocess plumbing left the graph for `harnais/host_exec.py`, and a
test now forbids the Harnais to import a graph.

### v3.24 — a unit domain

A fourth collector, and no new command. `busctl ListUnits` had run on
every `sysadmin` turn since August and `_parse_busctl_units` kept the
name and dropped the rest, though the fourth field of the reply is
`active_state`. `harnais/collectors/units.py` reports the units that are
neither `active` nor `inactive` -- `failed`, and `activating
(auto-restart)`, which is what a restart loop looks like from outside --
and a `failed_count`, emitted even at zero, because zero failures
observed is an answer and no facts is not. Not the whole list: 522 units
on the Deck, half of them `.device` noise, would spend the budget saying
nothing is wrong. It is not V2 either: no Host Model, no persistence, no
correlation. What justified it was one question, "no errors on my
Deck?", answered without a single unit state in its context.

CI found what the development machine could not see. The fixtures that
simulate "every collector is down" did not patch the new one, so it ran
the real `busctl`. The Flatpak sandbox where the suite is developed has
none: it failed, the fixture looked complete, the suite was green. A CI
image has `busctl`, it answered, and four refusal tests fell for a
reason unrelated to what they test.
`test_the_blind_fixture_really_blinds_every_collector` now checks the
registry of collectors rather than a context, so a collector added
later fails by name. The first check of the fix was worthless: a fake `busctl` on
`PATH` is never seen, because `subprocess_env()` pins `PATH`.

What it cannot see is its own proxies. `forge-dbus-proxy.sh` exposes the
system bus and both proxies are `systemd --user` units, so `ListUnits`
never lists them: "why is forge-podman-ro-proxy not working?" is the one
question `sysadmin` cannot answer, and the one that would have caught
the three-day outage of 2026-09-11. Closing it takes a second filtered
proxy on the session bus, which widens what Forge reaches into the
user's systemd manager -- a decision, not an oversight.
[deploy/README.md](../deploy/README.md) says so.

---

[← Documentation index](README.md) · [← Project README](../README.md)
