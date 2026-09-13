#!/usr/bin/env python3
"""
How much retrieved text can the loaded model synthesize before it
stops answering?

WHY THIS EXISTS

On 2026-09-13, `research` answered a question about bœuf bourguignon
with this, on screen, to the user:

    {
      "tool": "chat",
      "content": "..."
    }

Everything upstream had worked. The router picked `research`, the
search returned five results, three pages were fetched. The synthesis
call is where it went. A second failure the same day, on a question
whose fetch returned ONE off-topic page, produced a nested routing
decision instead: {"tool": "web_search", ...}.

Neither is a bug in the chain. Sampling a synthesis under the router's
grammar is deliberate and measured (providers/llama_cpp.py, 2026-08-21),
and try_unwrap_router_json is the seam that makes it safe -- in every
case that works, the log says "wrapped a substantive answer ...
unwrapped it" and the user sees prose. What failed is that the model
put nothing substantive in `content`, so the unwrap correctly refused
to dress up "..." as an answer and showed the raw object instead.

The question this harness answers is the one nobody could answer that
day: how much fetched text can this model carry before that happens?
`RESEARCH_FETCH_TOP_N` defaults to 3, and that default was tuned
around a 9B. Nothing said what it should be for anything else.

WHAT IT MEASURES, AND AGAINST WHAT

One axis: the size of the fetch block, in tokens of final prompt. The
query, the template, the grammar, the cleaner and the language line
are production's own, imported rather than rebuilt -- a harness that
reimplements them measures a path production does not use, which has
cost this repo a finding twice.

The verdict is what the USER WOULD SEE, i.e. the output of
research._clean_synthesis_response, not the model's raw text. Those
differ in exactly the case under test: raw JSON is a failure, the same
JSON unwrapped into three paragraphs is a success, and only the cleaner
knows which happened.

WHY REPEATS ARE NOT THE SAMPLE SIZE

At temperature 0 in one llama-server process, the same prompt returns
the same tokens. Repeating it measures nothing. (Across a RESTART it
can differ -- that is the h02 finding, and it is why --repeat exists at
all, for an operator deliberately bouncing the server between passes.)

So the sample is the SUBJECTS. Four of them, on different topics, with
differently-shaped pages. A ceiling is only worth quoting if the
subjects agree on roughly where it is; the report prints the spread and
says so when they do not.

WHY THE PAGES ARE CACHED

Fetched live, this harness would drift with the web: a page that gains
a cookie banner changes the measurement without anything about Forge
changing. `--capture` fetches once into bench/.research_ceiling_cache
.json (gitignored, like every other result file here) and every later
run reads it. Two runs weeks apart then differ only by what they were
meant to differ by. Recapture deliberately, not by accident.

A GRID, NOT A BISECTION

Bisection would cost four calls per subject instead of six, and assumes
the pass/fail boundary is monotonic. Nothing establishes that. The grid
shows the shape, including a subject that fails at one size and passes
at a larger one -- which would mean the ceiling is not what is being
measured, and is worth knowing before a number from here is used to set
a default.

Every row costs one completion, ten to thirty seconds of it, so the
table is printed as it is produced rather than at the end -- stdout is
reconfigured line-buffered below. Left block-buffered, a run that is
working is indistinguishable from one that has hung for five minutes.
That is not hypothetical: it happened on this file's first run against
a second model.

READING IT

  PASS     the user sees prose. What you want.
  RAW      raw JSON reached the screen. The failure from 2026-09-13.
  SHORT    unwrapped, but under 200 chars -- an answer in shape only.
  ERROR    the cleaner replaced it with [error]..., i.e. Forge caught it.
           Better than RAW, still not an answer.

The ceiling for a subject is the largest size that passed with nothing
failing below it. `--out` writes the whole table as JSON, same
convention as router_ab.py, so two models can be compared the way
CLAUDE.md describes: capture, swap the model, run again, diff by hand.

    bench/in_container.sh research_ceiling --capture
    bench/in_container.sh research_ceiling --out /tmp/ceiling-9b.json

RESULT, Deck, 2026-09-13, CloudSurf-4B-FC.Q4_K_M
-------------------------------------------------
There is no ceiling to find with this model. THREE OF FOUR SUBJECTS
FAIL AT THE SMALLEST POINT -- 846 to 866 tokens, one page capped at
1500 characters, which is the least this graph can ever ask of a
model. The failure is identical every time and costs 26 to 28 tokens:

    {"tool": "web_search", "content": "recette bœuf bourguignon"}

That is a ROUTING DECISION, emitted where a synthesis was asked for.
sqlite fails differently and worse: it unwraps to the user's own
question, copied back -- the h02 defect, in the one place where the
answer was sitting in the prompt.

Only `podman` synthesizes, and it does so up to 2569 tokens before
failing at 3402. So the harness can say PASS; the other three are not
an artefact of it.

Read against the two arms this replaced, that is the whole finding.
An earlier ad-hoc version reported 4/4 correct on the same prompts --
it ran under the 677-character FALLBACK grammar (no load_tools) and
posted its own payload rather than going through llama_cpp.call, so
the chat template never applied. Both faults made the model's job
easier than production makes it. Neither RESEARCH_FETCH_TOP_N nor
RESEARCH_FETCH_CHARS_PER_RESULT is the answer here, because the
smallest setting fails too: a 4B is not doing this task under this
graph's prompt and this grammar.

WHAT THIS DOES NOT SAY. Nothing about Qwen3.5-9B, which was not
loaded. Run the same command with it to get the number this file was
written to produce -- the cache makes the two runs differ only by the
model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Line-buffered, because every row of the table below costs one
# completion and this is routinely run redirected to a file or through
# in_container.sh. Block-buffered, a run that is working is
# indistinguishable from one that has hung for five minutes.
sys.stdout.reconfigure(line_buffering=True)

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / "src", _HERE.parent / "src", Path("src")):
    if (_candidate / "forge").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from forge import lang
from forge.config import LLAMA_CPP_URL, LLM_MODEL
from forge.context_info import today_line
from forge.graphs import research
from forge.providers import llama_cpp
from forge.router.grammar import build_router_grammar
from forge.tokens import estimate_tokens
from forge.tools import registry, web_fetch

CACHE = _HERE / ".research_ceiling_cache.json"

#: Four subjects, not one. The sample size of this harness is the
#: subject count -- see the header on why repeats are not it. Chosen to
#: differ in the way that matters here: a recipe is a list, a
#: comparison is prose, a version question has one fact in it and a
#: lot of chaff around it.
SUBJECTS = {
    "bourguignon": {
        "query": "recette du bœuf bourguignon",
        "urls": [
            "https://cuisine.journaldesfemmes.fr/recette/346736-boeuf-bourguignon",
            "https://www.marmiton.org/recettes/recette_boeuf-bourguignon_18889.aspx",
            "https://marchepernoud.com/blog/la-recette-facile-du-boeuf-bourguignon/",
        ],
    },
    "wireguard": {
        "query": "différences entre WireGuard et OpenVPN",
        "urls": [
            "https://www.wireguard.com/",
            "https://openvpn.net/community-resources/how-to/",
            "https://en.wikipedia.org/wiki/WireGuard",
        ],
    },
    "sqlite": {
        "query": "quand utiliser SQLite plutôt que PostgreSQL",
        "urls": [
            "https://www.sqlite.org/whentouse.html",
            "https://www.sqlite.org/about.html",
            "https://www.postgresql.org/about/",
        ],
    },
    "podman": {
        "query": "qu'apporte la dernière version stable de Podman",
        "urls": [
            "https://docs.podman.io/en/latest/",
            "https://podman.io/get-started",
            "https://en.wikipedia.org/wiki/Podman",
        ],
    },
}

#: The grid, in the two settings production actually has. NOT an
#: abstract size: fetch_node caps each page at
#: RESEARCH_FETCH_CHARS_PER_RESULT and keeps RESEARCH_FETCH_TOP_N of
#: them, so a prompt of 5000 tokens is not something Forge can build
#: with the defaults, and measuring one would answer a question nobody
#: can act on. The first ad-hoc version of this measurement did exactly
#: that.
#:
#: Two axes crossed only at the default, rather than a full product:
#: 3x4 points per subject is 48 calls, and the pair that matters is
#: "which single setting do I lower", not their interaction.
DEFAULT_TOP_N = 3
DEFAULT_CHARS = 1500
TOP_N_AXIS = (1, 2, 3)
CHARS_AXIS = (1000, 1500, 2500, 4000)

PASS, RAW, SHORT, ERROR = "PASS", "RAW", "SHORT", "ERROR"

#: Below this, an unwrapped answer is an answer in shape only. Same
#: number research's own cleaner treats as substantive.
_MIN_ANSWER_CHARS = 200


def capture(only: str | None) -> dict:
    """Fetch every subject's pages once, through Forge's own web_fetch."""
    store = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    for name, spec in SUBJECTS.items():
        if only and name != only:
            continue
        pages = []
        for url in spec["urls"]:
            try:
                content = web_fetch.run(url)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:12} FAILED {url} -- {e}")
                continue
            if not content or content.startswith("[error]"):
                print(f"  {name:12} FAILED {url} -- {str(content)[:60]}")
                continue
            pages.append({"url": url, "content": content})
            print(f"  {name:12} {len(content):6} chars  {url}")
        if pages:
            store[name] = {"query": spec["query"], "pages": pages}

    CACHE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    print(f"\ncaptured to {CACHE}")
    return store


def build_prompt(query: str, pages: list[dict], top_n: int, chars: int) -> str:
    """
    Production's synthesis prompt, built the way fetch_node builds it.

    The two operations are the ones in graphs/research.py: keep the
    first *top_n* pages, cap each at *chars* characters with the same
    ellipsis. Reproduced here rather than imported because fetch_node
    reaches the network to get them -- but nothing else about the
    prompt is rebuilt: the template, the language line and the cleaner
    are production's own objects.
    """
    kept = []
    for p in pages[:top_n]:
        content = p["content"]
        if len(content) > chars:
            content = content[:chars].rstrip() + "…"
        kept.append({"url": p["url"], "content": content})

    search_block = "\n".join(
        f"{i}. {p['url']} -- {p['content'][:150]}" for i, p in enumerate(kept, 1)
    )
    fetch_block = "\n\n".join(f"[{p['url']}]\n{p['content']}" for p in kept)

    prompt = research._SYNTHESIS_PROMPT.format(
        today_line=today_line(),
        query=query,
        search_block=search_block,
        fetch_block=fetch_block,
    )
    # Appended exactly as _synthesize_node does, and for the same
    # reason: last position, and only when forge.lang is sure.
    return prompt + lang.line_for(query)


def verdict(raw: str) -> tuple[str, str]:
    """What the user would see, and a fragment of it for the report."""
    shown = research._clean_synthesis_response(raw)
    head = " ".join(shown.split())[:54]

    if shown.startswith("[error]"):
        return ERROR, head
    if shown.lstrip().startswith("{"):
        return RAW, head
    if len(shown) < _MIN_ANSWER_CHARS:
        return SHORT, head
    return PASS, head


def ask(prompt: str, grammar: str) -> tuple[str, int, float]:
    """
    One completion, through the provider production uses.

    llama_cpp.call rather than a hand-rolled requests.post, so the
    chat-template framing (LLAMA_CPP_APPLY_TEMPLATE) applies here
    exactly as it does in a real turn. A harness that posts its own
    payload measures a prompt production never sends -- which is the
    other half of the 2026-09-13 mistake this file's header describes.
    """
    started = time.monotonic()
    completion = llama_cpp.call(LLAMA_CPP_URL, LLM_MODEL, prompt, grammar=grammar)
    ms = (time.monotonic() - started) * 1000
    return completion.text, completion.usage.completion_tokens or 0, ms


def ceiling_for(rows: list[dict]) -> int | None:
    """
    Largest prompt that passed with nothing failing below it.

    Deliberately not "largest that passed": a subject that fails at
    3000 and passes at 4000 has not got a ceiling at 4000, it has
    something else going on, and quoting the larger number would hide
    exactly the case worth looking at.
    """
    ceiling = None
    for row in sorted(rows, key=lambda r: r["tokens"]):
        if row["verdict"] != PASS:
            break
        ceiling = row["tokens"]
    return ceiling


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--capture", action="store_true", help="fetch the pages, then stop")
    ap.add_argument("--only", choices=sorted(SUBJECTS), help="one subject")
    ap.add_argument("--out", help="write the full table as JSON")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="re-ask each prompt N times. Pointless within one server process "
        "at temperature 0 -- see the header. For an operator restarting "
        "llama-server between passes.",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="disable cache_prompt for this run, to rule out an ordering effect "
        "between sizes. Much slower: every prompt is prefilled from scratch.",
    )
    ap.add_argument("--dry", action="store_true", help="print sizes, call nothing")
    args = ap.parse_args()

    # Not optional and not cosmetic: TOOLS is filled by load_tools(),
    # not at import, so without this available_tools() is empty and
    # build_router_grammar() returns the fallback -- 677 characters
    # against production's 1005. The arms are then constrained
    # differently from production and the whole table says nothing.
    # This exact fault produced a discarded measurement on 2026-09-13.
    if args.no_cache:
        # Set on the module the provider actually reads, not on
        # forge.config: llama_cpp.py does `from forge.config import
        # LLAMA_CPP_CACHE_PROMPT`, which copies the value at import.
        llama_cpp.LLAMA_CPP_CACHE_PROMPT = False

    registry.load_tools()
    tools = sorted(registry.available_tools())
    if set(tools) <= {"chat", "code"}:
        print(f"tools enabled: {tools}\n")
        print(
            "REFUSING TO RUN: this looks like a fallback tool set, not a real\n"
            "deployment. Set ENABLED_TOOLS (or run this with the same\n"
            "environment as the container) so the grammar under test is the\n"
            "grammar actually served."
        )
        return 1

    if args.capture:
        capture(args.only)
        return 0

    if not CACHE.exists():
        print(f"no captured pages at {CACHE} -- run with --capture first.")
        return 1
    store = json.loads(CACHE.read_text(encoding="utf-8"))

    grammar = build_router_grammar()
    print(f"model: {LLM_MODEL}   tools: {len(tools)}   grammar: {len(grammar)} chars")
    print(f"cache_prompt: {not args.no_cache}   repeat: {args.repeat}\n")

    results = {}
    for name, data in store.items():
        if args.only and name != args.only:
            continue
        have = len(data["pages"])
        print(f"{name}  ({data['query']})   {have} page(s) captured")
        print(
            f"  {'top_n':>5} {'chars':>5} {'tokens':>7}  "
            f"{'verdict':8} {'gen':>5} {'ms':>6}  answer"
        )

        # The chars axis runs at whatever top_n this subject can build.
        # Pinning it to DEFAULT_TOP_N would delete the entire axis for a
        # subject whose third fetch failed -- and a failed fetch is
        # normal here, not exceptional: one of these URLs 404'd the
        # first time this was captured.
        chars_at = min(DEFAULT_TOP_N, have)
        points = {(n, DEFAULT_CHARS) for n in TOP_N_AXIS if n <= have}
        points |= {(chars_at, c) for c in CHARS_AXIS}

        rows = []
        for top_n, chars in sorted(points):
            prompt = build_prompt(data["query"], data["pages"], top_n, chars)
            tokens = estimate_tokens(prompt)
            if (top_n, chars) == (DEFAULT_TOP_N, DEFAULT_CHARS):
                mark = " <- deployed"
            elif (top_n, chars) == (chars_at, DEFAULT_CHARS):
                mark = f" <- deployed, but only {have} page(s) here"
            else:
                mark = ""

            if args.dry:
                print(f"  {top_n:>5} {chars:>5} {tokens:>7}  (dry){mark}")
                rows.append(
                    {"top_n": top_n, "chars": chars, "tokens": tokens, "verdict": None}
                )
                continue

            seen = []
            for _ in range(args.repeat):
                raw, gen, ms = ask(prompt, grammar)
                v, head = verdict(raw)
                seen.append((v, head, gen, ms))

            # Worst verdict of the repeats: one RAW in three is still a
            # size this model cannot be trusted at.
            order = {PASS: 0, SHORT: 1, ERROR: 2, RAW: 3}
            v, head, gen, ms = max(seen, key=lambda s: order[s[0]])
            print(
                f"  {top_n:>5} {chars:>5} {tokens:>7}  "
                f"{v:8} {gen:>5} {ms:>6.0f}  {head}{mark}"
            )
            rows.append(
                {
                    "top_n": top_n,
                    "chars": chars,
                    "tokens": tokens,
                    "verdict": v,
                    "gen": gen,
                    "ms": round(ms),
                    "verdicts": [s[0] for s in seen],
                }
            )

        results[name] = rows
        if not args.dry:
            c = ceiling_for(rows)
            print(f"  ceiling: {c if c else 'FAILED AT THE SMALLEST SIZE'}\n")

    if args.dry:
        return 0

    ceilings = {n: ceiling_for(r) for n, r in results.items()}
    found = [c for c in ceilings.values() if c]

    print("=" * 66)
    print(f"AT THE DEPLOYED SETTING (top_n={DEFAULT_TOP_N}, chars={DEFAULT_CHARS})")
    deployed = {}
    for name, rows in results.items():
        row = next(
            (
                r
                for r in rows
                if r["top_n"] == DEFAULT_TOP_N and r["chars"] == DEFAULT_CHARS
            ),
            None,
        )
        if row is None:
            print(f"  {name:14} not reached (fewer pages captured than top_n)")
            continue
        deployed[name] = row["verdict"]
        print(f"  {name:14} {row['verdict']:8} at {row['tokens']} tokens")

    bad = [n for n, v in deployed.items() if v != PASS]
    if bad:
        print(
            f"\n  {len(bad)} of {len(deployed)} subjects fail as deployed: "
            f"{', '.join(bad)}"
        )
        print("\n  LOWER ONE OF THE TWO. Read each failing subject's table above:")
        print("    - if it passes at a smaller top_n, set RESEARCH_FETCH_TOP_N")
        print("    - if it passes at a smaller chars, set")
        print("      RESEARCH_FETCH_CHARS_PER_RESULT")
        print("    - if it fails at every point, the synthesis is not working")
        print("      here at all and neither setting is the answer.")
    elif deployed:
        print("\n  Every subject passes as deployed. Nothing to change --")
        print("  and note what that means if a real turn still fails: the")
        print("  cause is something this harness holds fixed (the fetched")
        print("  pages themselves, or the search snippets above them).")

    ceilings = {n: ceiling_for(r) for n, r in results.items()}
    found = [c for c in ceilings.values() if c]
    print("\n" + "-" * 66)
    print("LARGEST PROMPT THAT PASSED, per subject (tokens)")
    for name, c in ceilings.items():
        print(f"  {name:14} {c if c else 'none -- failed at the smallest point'}")

    if len(found) >= 2:
        lo, hi = min(found), max(found)
        print(f"\n  median {int(statistics.median(found))}   spread {lo}-{hi}")
        if hi > lo * 2:
            print(
                "\n  THE SUBJECTS DISAGREE (spread is more than 2x). A single\n"
                "  number from this run would average two different behaviours\n"
                "  -- read the tables above before quoting one."
            )
    elif not found:
        print(
            "\n  Every subject failed at its smallest point. That is not a\n"
            "  ceiling, it is the synthesis not working here -- check the model\n"
            "  and the grammar length printed at the top before reading on."
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": LLM_MODEL,
                    "tools": tools,
                    "grammar_chars": len(grammar),
                    "repeat": args.repeat,
                    "cache_prompt": not args.no_cache,
                    "subjects": results,
                    "ceilings": ceilings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
