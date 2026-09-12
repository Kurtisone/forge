# Development

Tests, CI, and the measurement harnesses. The harnesses matter more than
they look: on this codebase, reasoning about what a prompt or a token
*should* do has lost to measurement often enough to be a rule.

## Continuous Integration

Every push to `main` and every PR targeting it runs, via GitHub Actions
(`.github/workflows/ci.yml`):

```bash
ruff check .
ruff format --check .
pytest -v
```

Same commands locally, after the same setup the workflow does:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

The editable install is what makes `forge` importable from the repo
root; without it `pytest -v` fails at collection with
`ModuleNotFoundError: No module named 'forge'`.
`ruff format --check` is a separate gate from `ruff check` and fails on
formatting alone -- running only the latter locally will let a patch
through that CI then rejects.

## Running the tests

```bash
PYTHONPATH=src pytest -q
```

One thing about that command is load-bearing and does not look it: the
directory holding `pytest` has to be on your **PATH**, not merely the
interpreter you invoke. `tools/test.py` resolves its runner with
`shutil.which()` against the real process PATH (deliberately — the
subprocess itself gets a minimal environment, and a runner installed
somewhere other than `/usr/bin` would be unreachable otherwise), so
running the suite as `…/venv/bin/python -m pytest` without activating
the venv fails six tests in `tests/test_test_tool.py` and the failures
look exactly like a regression in the tool. Activate the venv, or:

```bash
PATH=…/venv/bin:$PATH PYTHONPATH=src …/venv/bin/python -m pytest -q
```

The suite writes about 540 MB to `TMPDIR` — one sqlite-vec store per
test that needs one — and pytest keeps the last three runs. On a box
where `/tmp` is a small tmpfs (1.5 GB on the Deck), three full runs
fill it and the suite starts failing in bulk on `database or disk is
full`, which also looks like anything but its cause. `rm -rf
/tmp/pytest-of-$USER` between series.

### The web UI's renderer is executed, not read

Three test files say the same thing in their docstrings -- "this suite
has no JS runtime, so these are assertions on the source; a tripwire,
not a proof". That was true of CI and never true of this machine, which
has `deno`. `tests/test_ui_rendering.py` pulls `formatContent` and its
helpers straight out of `index.html` and runs them.

It found a bug the first time it ran. `_emphasis like this_` was
reaching the screen with its underscores showing, because the UI
implements `**bold**` and `*em*` and nothing else -- and both footers
`graphs/research.py` appends were written in the syntax it does not
have. One of them had been wrong since it was written.

The functions are EXTRACTED rather than copied into a fixture. A second
copy is a copy that drifts, and drift in a renderer is invisible until
someone reads their own output and finds punctuation in it.

Skipped where `deno` is absent, which includes CI today. A test that
runs on one machine and skips on another is worth more than no test.

## The measurement harnesses

`bench/` is not a second test suite. Tests pin behaviour that must not
change; these answer questions the test suite cannot ask — what a
distance is on *this* store, what a prompt costs on *this* box, whether
a model can make a judgement at all. Most of them end in a number that
went into a default, or in a finding that killed a mechanism.

| Harness | What it answers | Needs |
|---|---|---|
| `router_ab` | What the router prompt costs and what it decides (see below) | server, two checkouts |
| `recall_distance` | What distance a good memory hit sits at here | server + embeddings |
| `instruct_prefix` | Whether the embedding model's query instruction helps on this store | store copy |
| `rag_dilution` | What burying a sentence in a compacted block costs | store copy |
| `recall_expansion` | What asking again in other words rescues, and what it lets in | store copy + model |
| `rag_hybrid` | Which channel reaches which entry, and what the words drag in | store copy |
| `rag_hot_tier` | What the hot block answers, costs, and makes redundant | store copy |
| `rag_aggregate` | What the aggregation pass would fold and what it would refuse — **free and deterministic since v3.19**, writes only under `--apply` | store copy |
| `sysadmin_verdict` | Whether the model can tell that a log block does not answer the question | model |
| `no_think_ab` | Whether `/no_think` still does anything on the synthesis prompts | model |
| `prose_grammar_ab` | What giving the graph syntheses their own grammar costs | model |

What each of them found, and which default it settled, is in [The measurement
record](campaigns.md) — read it before re-running one, because several of these
questions are closed and the page says so in the first line of each section.

Everything that reads the store runs through one script:

```bash
bench/in_container.sh rag_hybrid --db /tmp/real_copy.db \
    --hit "Tu peux me lister mon matériel ?" --expect 307 \
    --miss "Comment s'appelle mon chat ?"
```

`in_container.sh` copies this checkout and a **fresh copy of the store**
into the container and runs one harness there with `PYTHONPATH` pointing
at the copy. Both halves matter. Without the copy, a benchmark is one
keystroke away from measuring — or writing to — production. Without the
`PYTHONPATH`, `from forge import rag` resolves to the image's deployed
code rather than the checkout you just copied in, which is precisely the
case you reach for a harness in.

`sysadmin_verdict` is the one harness with no store behind it: the log
blocks it asks about are written, because `traces.jsonl` records the
*source* of a collection and never its content. It says so at the top,
for the reason `bench/_harness.py` has a `PLACEHOLDER` check — a harness
producing a confident verdict from its own boilerplate has happened here
twice.

## Prompt Cache & Routing A/B (v3.12)

`bench/router_ab.py` measures what the router prompt costs and what it
decides. It exists because the two failure modes it covers are invisible
from the test suite: a prompt-cache regression has no functional symptom
at all (every answer stays correct, runs just get slower), and a routing
regression is masked by the GBNF grammar, which guarantees the output
*shape* whatever the model picks.

Three measurements, deliberately separate:

| | what it answers | needs a server |
|---|---|---|
| `prefix` | how many characters diverge between consecutive prompts | no |
| `bench` | prompt-processing time on a growing conversation | yes |
| `routing` | which tool gets picked, across 29 fixtures | yes |

`prefix` is pure string arithmetic and fully deterministic, so it is the
one to trust when the other two disagree. A prompt that is a strict
prefix of the next one continues from llama-server's live slot state; an
insertion anywhere above forces a rewind to the last checkpoint, and past
a certain depth a full recompute.

```bash
# no llama-server needed
python bench/router_ab.py run --offline --out before.json

# full run, against the configured provider
python bench/router_ab.py run --out after.json
python bench/router_ab.py compare --before before.json --after after.json
```

The two arms of a comparison are two checkouts -- the harness never
rebuilds the old prompt itself. Run it once per branch, then compare.
It refuses to start on a fallback tool set, since `ENABLED_TOOLS` decides
what the prompt contains and an A/B across two different tool sets
compares two prompts rather than two layouts.

Read `agreement` rather than the pass counts: on 29 fixtures a one- or
two-fixture difference is noise, and a changed decision is worth opening
by hand even when it changed from fail to pass.

---

[← Documentation index](README.md) · [← Project README](../README.md)
