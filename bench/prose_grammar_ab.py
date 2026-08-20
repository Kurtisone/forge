#!/usr/bin/env python3
"""
What does giving the graph syntheses their own grammar actually cost?

Until this branch, every graph called call_llm(prompt) with no
grammar, and providers/llama_cpp._grammar_for() read that as "use the
router's". So four synthesis prompts that spend paragraphs asking for
plain text were sampled under a grammar admitting only
{"tool":...,"content":...}. That is why recall came back wrapped on
three runs out of three, and why sysadmin and research have logged
"model wrapped a substantive answer in router-style JSON" since v3.11.

The fix is forge/prose_grammar.py. It has one real risk, and this
harness exists to measure exactly that risk and nothing else:

  Under the ROUTER grammar, generation stopped STRUCTURALLY. The
  closing brace ended the object and the sampler had nowhere to go --
  16 to 17 tokens on a routing call, and a hard ceiling on a synthesis
  one. Under a prose grammar the tail is starred, so the only thing
  that ends generation is the model emitting EOS. Nothing structural
  bounds it any more. At ~90 ms/token measured on this box, a model
  that starts padding is expensive, and it would show up as "Forge got
  slower" with no other symptom -- the same shape as the
  LLAMA_CPP_CACHE_PROMPT=false problem, which survived weeks.

So the column that decides whether this branch ships as-is is GEN
(completion tokens). ENV (a JSON envelope reaching the user) is the
column that says whether the fix worked at all.

Both arms come from ONE checkout and one prompt each. The only thing
that differs is the grammar argument, so the arms cannot drift for an
unrelated reason. Prompts and cleaners are the graphs' own, imported
rather than copied: a harness that reimplements production's cleaning
measures a path production does not use, which has already cost a
finding on this repo twice.

    podman exec forge sh -c 'rm -rf /tmp/arm && mkdir -p /tmp/arm'
    podman cp src forge:/tmp/arm/
    podman cp bench/prose_grammar_ab.py forge:/tmp/arm/
    podman exec -it forge python /tmp/arm/prose_grammar_ab.py --repeat 2

The rm -rf is not cosmetic: podman cp MERGES into an existing
directory instead of replacing it, so without it the run is a mixture
of two checkouts.

Columns:

  GEN   completion tokens. THE number. Compare arms per row, not rows
        to each other -- the four prompts are different sizes.
  CAP   the run hit LLAMA_CPP_N_PREDICT, i.e. it was cut off rather
        than finishing. Any CAP in the prose arm is a stop-and-think.
  ENV   the answer reaching the user still starts with {"tool" --
        under the router arm this is the bug; under the prose arm it
        would mean the grammar was dropped (check the log for
        "grammar rejected before sending").
  MS    wall milliseconds for the call.
  ECHO  the answer copied the prompt's own GOOD ANSWER example. Not
        what this harness is for, but free to check and it has caught
        a regression here before (see no_think_ab.py).

Read GEN first. A prose arm within roughly 1.5x of the router arm is
the expected result: the router arm's number is inflated by the JSON
scaffolding it has to emit, so a modest increase is not padding. A 3x
or a CAP is padding, and the answer is a length bound in the grammar,
not a sentence added to the prompt.
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

from forge import prose_grammar
from forge.config import LLAMA_CPP_N_PREDICT, LLAMA_CPP_URL, LLM_MODEL
from forge.context_info import today_line
from forge.graphs import recall, research, review, sysadmin
from forge.providers import llama_cpp
from forge.tools import registry

# Populating the registry is not optional and not cosmetic. TOOLS is
# filled by load_tools() when the API starts, NOT at import -- so in a
# fresh `podman exec python script.py` process available_tools()
# returns [] and build_router_grammar() falls back to _FALLBACK_TOOLS,
# giving the router arm a grammar six times smaller than the real one.
# Measured against the wrong baseline, this whole harness says nothing.
registry.load_tools()

THINK_OPEN = re.compile(r"<think>|<thinking>|◁think▷")

ECHOES = {
    "recall": ["exemple-hôte", "modèle-fictif"],
    "review": ["Naming and structure are otherwise clear", "raising ValueError"],
    "research": ["Plusieurs sorties majeures", "hausse des ventes"],
    "sysadmin": ["exemple-service.service", "manquant.conf"],
}

CLEANERS = {
    "recall": recall._clean_synthesis_response,
    "review": review._clean_review_response,
    "research": research._clean_synthesis_response,
    "sysadmin": sysadmin._clean_diagnosis_response,
}

# Which grammar each graph actually ships with. recall asks for one
# sentence and gets the strict form; the other three quote logs and
# web pages, so only their first character is constrained.
SHIPPED = {
    "recall": prose_grammar.SENTENCE,
    "review": prose_grammar.PROSE,
    "research": prose_grammar.PROSE,
    "sysadmin": prose_grammar.PROSE,
}

# The same four fixtures as no_think_ab.py, deliberately. Small and
# self-contained: the question is generation length on a normal
# answer, not whether the model can handle a hard case. Using the same
# ones means the two harnesses' numbers can be read side by side.
CASES = {
    "recall": lambda: recall._SYNTHESIS_PROMPT.format(
        query="Quel port utilise le serveur ?",
        entries_block="- PORT = 8080 dans la config\n- DEBUG était à True",
        language_line="",
    ),
    "review": lambda: review._REVIEW_PROMPT.format(
        today_line=today_line(),
        filename="add.py",
        question="Que peut-on améliorer ?",
        content="def add(a, b):\n    return a - b\n",
        test_section="",
    ),
    "research": lambda: research._SYNTHESIS_PROMPT.format(
        today_line=today_line(),
        query="nouveautés de Qwen3.5",
        search_block="1. Qwen3.5 notes - https://example.org/q - hybrid attention",
        fetch_block="Qwen3.5 introduces gated DeltaNet layers and a 262k context.",
    ),
    "sysadmin": lambda: sysadmin._SYNTHESIS_PROMPT.format(
        today_line=today_line(),
        question="Pourquoi searxng redémarre en boucle ?",
        source="podman:searxng",
        log_block="OSError: [Errno 98] Address already in use\nexited with code 1",
    ),
}


def run_arm(graph: str, arm: str, prompt: str) -> dict:
    # llama_cpp.call rather than call_llm, because completion_tokens is
    # the measurement and call_llm returns only the text. This is the
    # same function call_llm makes for this provider; what is skipped
    # is metrics recording, which nothing here reads.
    #
    # grammar=None is not "no grammar": _grammar_for() turns it into
    # the router's. That IS the arm.
    grammar = None if arm == "router" else SHIPPED[graph]

    started = time.monotonic()
    result = llama_cpp.call(LLAMA_CPP_URL, LLM_MODEL, prompt, grammar)
    elapsed = int((time.monotonic() - started) * 1000)

    answer = CLEANERS[graph](result.text)
    gen = result.usage.completion_tokens
    return {
        "graph": graph,
        "arm": arm,
        "gen": gen,
        "cap": gen is not None and gen >= LLAMA_CPP_N_PREDICT,
        "env": answer.lstrip().startswith('{"tool"'),
        "ms": elapsed,
        "think": bool(THINK_OPEN.search(result.text)),
        "echo": [f for f in ECHOES.get(graph, []) if f in answer],
        "answer": answer,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", choices=sorted(CASES))
    ap.add_argument("--dry", action="store_true", help="print prompts, call nothing")
    args = ap.parse_args()

    graphs = [args.only] if args.only else sorted(CASES)

    if args.dry:
        for g in graphs:
            print(f"===== {g} =====\n{CASES[g]()}\n")
            print(f"----- grammaire livrée -----\n{SHIPPED[g]}")
        return 0

    rows = []
    for g in graphs:
        prompt = CASES[g]()
        for _ in range(args.repeat):
            # Router arm first, every time. Both arms share a prompt,
            # so whichever runs second gets a warm KV prefix -- fixing
            # the order at least makes that bias constant instead of
            # alternating, and it favours the arm under test, which is
            # the direction that cannot flatter a false positive.
            for arm in ("router", "prose"):
                row = run_arm(g, arm, prompt)
                rows.append(row)
                print(
                    f"{row['graph']:9} {row['arm']:7} "
                    f"GEN={row['gen']!s:>4} "
                    f"CAP={'Y' if row['cap'] else '.'} "
                    f"ENV={'Y' if row['env'] else '.'} "
                    f"MS={row['ms']:>6} "
                    f"THINK={'Y' if row['think'] else '.'} "
                    f"ECHO={','.join(row['echo']) or '.'}"
                )
                print(f"          {row['answer'][:160]!r}")

    print("\n--- verdict ---")
    for g in graphs:
        arm_gen = {}
        for arm in ("router", "prose"):
            vals = [r["gen"] for r in rows if r["graph"] == g and r["arm"] == arm]
            vals = [v for v in vals if v is not None]
            arm_gen[arm] = sum(vals) / len(vals) if vals else None
        r, p = arm_gen["router"], arm_gen["prose"]
        ratio = f"{p / r:.2f}x" if r and p else "n/a"
        envs = sum(
            1 for x in rows if x["graph"] == g and x["arm"] == "prose" and x["env"]
        )
        caps = sum(
            1 for x in rows if x["graph"] == g and x["arm"] == "prose" and x["cap"]
        )
        print(
            f"{g:9} router={r} prose={p} ({ratio})"
            f"{'  ENVELOPE STILL PRESENT' if envs else ''}"
            f"{'  HIT N_PREDICT' if caps else ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
