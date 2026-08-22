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
