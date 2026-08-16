#!/usr/bin/env python3
"""
Does the /no_think prefix on the graph synthesis prompts still do
anything? Measured 2026-08-16: yes, and it stays.

router_ab.py does not cover this. It exercises the ROUTER prompt, which
never carried the prefix. The four prompts that do are the recall /
review / research / sysadmin synthesis steps, and what matters there is
answer quality, not routing and not ms/token.

VERDICT -- read this before removing the prefix again
-----------------------------------------------------
The prefix looked dead on paper: Qwen3.5 dropped the /think soft
switch, and the router GBNF grammar (applied to EVERY call, not just
routing) forces the first token to "{", so no reasoning block can be
emitted whatever the prompt says. Both true. Both irrelevant.

Removing it, on this model:

  recall    returned the prompt's own GOOD ANSWER example -- about
            hardware -- to a question about a port.
  review    returned its GOOD ANSWER example verbatim and end to end,
            missing the actual bug in the file it was given.
  research  padded, no copy.
  sysadmin  unaffected.

Twice each, deterministic. Whatever the prefix does at position 0, it
is not what its name says, and it measurably changes how this model
attends to the examples further down the prompt. Reasoning from what a
token MEANS to the model that defined it predicted the wrong answer
here; only the measurement got it right.

Both arms come from ONE checkout: the prompt as committed is `prefix`
(what ships), and `noprefix` is that same prompt with "/no_think\\n"
stripped off the front. No checkout swapping, so the arms cannot drift
for an unrelated reason.

Everything goes through forge.llm.call_llm, so both arms see production
conditions: same temperature, stop sequences, slot, and grammar.

    podman exec forge sh -c 'rm -rf /tmp/arm && mkdir -p /tmp/arm'
    podman cp src forge:/tmp/arm/
    podman cp bench/no_think_ab.py forge:/tmp/arm/
    podman exec -it forge python /tmp/arm/no_think_ab.py --repeat 2

The rm -rf is not cosmetic: podman cp merges into an existing directory
instead of replacing it, so without it the run is a mix of checkouts.

Use --dry to print both prompt variants and call nothing, --only to run
a single graph.

What the columns mean, in order of importance:

  ECHO     the answer contains a distinctive fragment of the prompt's
           own GOOD ANSWER example, i.e. the model copied the example
           instead of using the material it was given. The first
           version of this harness did not check for this and called
           the answer text "the one measurement a machine cannot score
           for you" -- which was wrong, and cost a finding: recall's
           copy came back with no flag at all, because a copied example
           contains no marker and reads like a real answer.
  think    a <think> block in the raw output. Never observed, in either
           arm, on any graph -- the grammar makes it impossible.
  gen      characters generated. A large jump is the softer version of
           the ECHO worry.
  answer   the cleaned text, printed WHOLE. The first version cut it at
           300 chars, which hid the tail -- exactly where a copied
           example sits, so the display was hiding the evidence it
           existed to show.
"""

import argparse
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / "src", _HERE.parent / "src", Path("src")):
    if (_candidate / "forge").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from forge.context_info import today_line
from forge.graphs.recall import _SYNTHESIS_PROMPT as RECALL
from forge.graphs.research import _SYNTHESIS_PROMPT as RESEARCH
from forge.graphs.review import _REVIEW_PROMPT as REVIEW
from forge.graphs.sysadmin import _SYNTHESIS_PROMPT as SYSADMIN
from forge.llm import call_llm
from forge.text_cleaning import strip_think_blocks, try_unwrap_router_json

PREFIX = "/no_think\n"
THINK_OPEN = re.compile(r"<think>|<thinking>|◁think▷")

# Distinctive fragments of each prompt's GOOD ANSWER example. One
# appearing in an answer means the model copied the example. recall's
# and sysadmin's are also enforced at runtime (_EXAMPLE_LEAK_FRAGMENTS
# in those modules); review's and research's are only watched here --
# their examples are ordinary English prose that a legitimate answer
# could plausibly reproduce, so they are not safe as runtime checks
# without a fictional rewrite first.
ECHOES = {
    "recall": ["exemple-hôte", "modèle-fictif"],
    "review": ["Naming and structure are otherwise clear", "raising ValueError"],
    "research": ["Plusieurs sorties majeures", "hausse des ventes"],
    "sysadmin": ["exemple-service.service", "manquant.conf"],
}

# One fixture per graph. Deliberately small and self-contained: the
# question is whether the model answers from the material it was given,
# not whether it can handle a hard case.
CASES = {
    "recall": lambda: RECALL.format(
        query="Quel port utilise le serveur ?",
        entries_block="- PORT = 8080 dans la config\n- DEBUG était à True",
    ),
    "review": lambda: REVIEW.format(
        today_line=today_line(),
        filename="add.py",
        question="Que peut-on améliorer ?",
        content="def add(a, b):\n    return a - b\n",
        test_section="",
    ),
    "research": lambda: RESEARCH.format(
        today_line=today_line(),
        query="nouveautés de Qwen3.5",
        search_block="1. Qwen3.5 notes - https://example.org/q - hybrid attention",
        fetch_block="Qwen3.5 introduces gated DeltaNet layers and a 262k context.",
    ),
    "sysadmin": lambda: SYSADMIN.format(
        today_line=today_line(),
        question="Pourquoi searxng redémarre en boucle ?",
        source="podman:searxng",
        log_block="OSError: [Errno 98] Address already in use\nexited with code 1",
    ),
}


def clean(raw: str, source: str) -> str:
    cleaned = strip_think_blocks(raw)
    unwrapped = try_unwrap_router_json(cleaned, source=source)
    return unwrapped if unwrapped is not None else cleaned


def run_arm(name: str, prompt: str, source: str) -> dict:
    started = time.monotonic()
    raw = call_llm(prompt)
    elapsed = int((time.monotonic() - started) * 1000)
    answer = clean(raw, source)
    return {
        "arm": name,
        "ms": elapsed,
        "gen": len(raw),
        "think": bool(THINK_OPEN.search(raw)),
        "echo": [f for f in ECHOES.get(source, []) if f in answer],
        "answer": answer,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", help="recall / review / research / sysadmin")
    ap.add_argument("--dry", action="store_true", help="print prompts, call nothing")
    args = ap.parse_args()

    cases = CASES if not args.only else {args.only: CASES[args.only]}
    failures = []

    for source, build in cases.items():
        shipped = build()
        if not shipped.startswith(PREFIX):
            print(
                f"!! {source}: prompt no longer starts with {PREFIX!r} -- see VERDICT"
            )
            return 2
        stripped = shipped[len(PREFIX) :]

        if args.dry:
            print(f"===== {source}: prefix =====\n{shipped}\n")
            print(f"===== {source}: noprefix =====\n{stripped}\n")
            continue

        print(f"\n===== {source} =====")
        # Alternate the arms rather than running one arm to completion:
        # a warm slot favours whichever went second, and alternating
        # spreads that bias across both instead of handing it to one.
        for i in range(args.repeat):
            for arm, prompt in (("prefix", shipped), ("noprefix", stripped)):
                r = run_arm(arm, prompt, source)
                flags = ""
                if r["think"]:
                    flags += " THINK!"
                    failures.append(f"{source}/{r['arm']}: reasoning block")
                if r["echo"]:
                    flags += f" ECHO!{r['echo']}"
                    failures.append(f"{source}/{r['arm']}: copied {r['echo']}")
                print(f"[{i}] {r['arm']:<8} {r['ms']:>6} ms  gen={r['gen']:>4}{flags}")
                # Whole answer, never truncated -- see the header.
                for line in r["answer"].splitlines() or [""]:
                    print(f"      {line}")

    shipping = [f for f in failures if "/prefix:" in f]
    if failures:
        print("\n!! " + "\n!! ".join(failures))
    if shipping:
        print("\n!! failures in the SHIPPING arm -- do not merge")
        return 1
    if failures:
        print("\n(noprefix-only failures are the expected result -- see VERDICT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
