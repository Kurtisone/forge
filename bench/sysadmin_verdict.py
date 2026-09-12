#!/usr/bin/env python3
"""
Can the model say whether a log block answers the question?

    bench/in_container.sh sysadmin_verdict
    bench/in_container.sh sysadmin_verdict --arm cite

WHY THIS EXISTS. graphs/sysadmin.py collects logs before anyone has
read the question -- it has to, the collection is what makes the
question answerable -- so the logs may simply not contain the answer.
The prompt says so, in as many words, and asks the model to say so
plainly. When it complies, the reply is prose: "Les logs fournis ne
contiennent aucune information explicite sur la cause du redémarrage."

That sentence is correct and invisible. forge/non_answer.py cannot
match it without a phrase list that would start dropping real answers,
and the run reports nothing, so the exchange is indexed as though it
were a diagnosis -- three of the real store's archived entries were
exactly that on 2026-09-12. The fix that suggests itself is a VERDICT
the code can read instead of a sentence it has to parse.

This harness is what that idea has to get past. Four ways of asking,
one model call each, against fixtures whose answer is known:

  plain   "Do these logs contain what is needed?" YES / NO
  negated the same question inverted, NO first in the grammar --
          the control for a model that agrees with whatever it is
          asked
  cite    "Which line answers it? Reply with its number, or NONE",
          the alternation built over the lines actually present so a
          fabricated citation is unsamplable
  reason  one bounded phrase, then " => ", then the verdict

Both error directions are counted, because they are not the same
mistake. A false NO refuses a diagnosis that was available. A false
YES is the status quo: a fluent paragraph built on logs about
something else, which is the failure this tool cannot recover from.

THE FIXTURES ARE FIXTURES. Every other harness in this directory runs
against a copy of the real store; this one cannot, because the log
blocks are gone -- traces.jsonl records the source of a collection and
never its content. So the blocks below are written to look like what
the real cases held, and the shapes come from runs that actually
happened: the SearXNG restart whose logs show a normal startup
(archived #24 and #27), and run #be385d16, where a routine llama.cpp
n_batch notice was read as evidence of "une assertion ou une erreur
interne NON AFFICHÉE".

The ground truth in the table is the author's. That is the weakness of
this harness and there is no version of it without one: deciding
whether a log block answers a question is the judgement under test.
"""

import argparse
import sys
import time

# --- The fixtures ----------------------------------------------------------

NORMAL_START = """\
2026-09-11 08:12:03,114 INFO:searx.webapp: starting webserver on http://0.0.0.0:8080/
2026-09-11 08:12:03,115 INFO:werkzeug: Press CTRL+C to quit
2026-09-11 08:12:04,882 INFO:searx.engines: 92 engines loaded
2026-09-11 08:12:05,001 INFO:searx.search.processors: processors initialized
2026-09-11 08:12:09,447 INFO:werkzeug: 10.89.0.4 - - [11/Sep/2026 08:12:09] "GET /healthz HTTP/1.1" 200 -"""

OUT_OF_MEMORY = """\
2026-09-11 08:11:57,003 WARNING:searx.engines: engine timeout: google
2026-09-11 08:11:58,441 INFO:searx.webapp: shutting down
[11223.884] Out of memory: Killed process 4412 (searxng-run) total-vm:2118344kB
2026-09-11 08:12:03,114 INFO:searx.webapp: starting webserver on http://0.0.0.0:8080/"""

ROUTINE_NOTICE = """\
llama_context: n_ctx_per_seq (8192) < n_ctx_train (32768) -- the full capacity will not be used
llama_context: CPU output buffer size = 0.58 MiB
main: server is listening on http://0.0.0.0:8082 - starting the main loop
srv update_slots: all slots are idle"""

CRASH = """\
llama_context: CPU output buffer size = 0.58 MiB
main: server is listening on http://0.0.0.0:8082 - starting the main loop
GGML_ASSERT(ggml_nelements(a) == ne0*ne1) failed
/app/ggml/src/ggml.c:3421: GGML_ASSERT failed
Aborted (core dumped)"""

CLEAN_KERNEL = """\
[    0.000000] Linux version 6.16.12-valve24.5-1-neptune-616
[    1.884113] systemd[1]: Detected architecture x86-64.
[    2.004221] systemd[1]: Reached target Multi-User System.
[   12.881004] wlan0: associated"""

ANOTHER_SERVICE = """\
2026-09-11 09:01:12 forge.api: POST /chat 200 in 16284ms
2026-09-11 09:02:44 forge.api: GET /history 200 in 4ms
2026-09-11 09:03:02 forge.rag: embedded 3 chunks in 221ms"""

#: (question, source, logs, whether the logs answer it)
CASES = [
    ("pourquoi searxng a redémarré ?", "podman logs searxng", NORMAL_START, False),
    ("pourquoi searxng a redémarré ?", "podman logs searxng", OUT_OF_MEMORY, True),
    (
        "pourquoi forge-embedding plante ?",
        "podman logs forge-embedding",
        ROUTINE_NOTICE,
        False,
    ),
    ("pourquoi forge-embedding plante ?", "podman logs forge-embedding", CRASH, True),
    # A clean journal DOES answer "are there errors?", with "no". The
    # case is here to separate "the logs do not answer" from "the
    # answer is negative", which is the distinction the whole idea
    # rests on.
    ("Des erreurs dans le journal ?", "journalctl -k", CLEAN_KERNEL, True),
    ("pourquoi searxng a redémarré ?", "podman logs forge", ANOTHER_SERVICE, False),
    ("quel port écoute searxng ?", "podman logs searxng", NORMAL_START, True),
    # Nothing at all. No model is needed to answer this one, which is
    # exactly why it is here: an arm that says YES on an empty block
    # is not reading the block.
    ("pourquoi searxng a redémarré ?", "podman logs searxng", "", False),
]

_PREAMBLE = """/no_think
These logs were gathered before anyone read the question below. They
may simply not contain what the question asks about.

Question: {question}

--- collected logs ({source}) ---
{logs}
--- end of collected logs ---
"""

_PLAIN = (
    _PREAMBLE
    + """
Do these logs contain what is needed to answer that question?
Answer YES or NO and nothing else."""
)

_NEGATED = (
    _PREAMBLE
    + """
Is the answer to that question MISSING from these logs?
Answer NO or YES and nothing else."""
)

_CITE = """/no_think
These logs were gathered before anyone read the question below. They
may simply not contain what the question asks about.

Question: {question}

--- collected logs ({source}), one numbered line each ---
{logs}
--- end of collected logs ---

Which single line answers that question? Reply with its number, or
NONE if no line does. Reply with the number or NONE and nothing else."""

_REASON = (
    _PREAMBLE
    + """
First say in one short phrase what these logs actually show, then
" => " and YES if they contain what is needed to answer the question,
NO if they do not."""
)

#: Bounded, because free prose has no terminator and runs to
#: n_predict -- the lesson tests/test_graph_grammar.py carries. The
#: apostrophe is NOT backslash-escaped: llama.cpp rejects `\\'` inside a
#: character class and answers 400 to the whole call.
_WORD = "[a-zA-Z0-9éèêàçùûôîïœ_/.,:;()'-]+"
_REASON_GRAMMAR = (
    'root ::= verdict-reason " => " ("YES" | "NO")\n'
    f"verdict-reason ::= verdict-word{' (" " verdict-word)?' * 24}\n"
    f"verdict-word ::= {_WORD}\n"
)


def _numbered(logs: str) -> str:
    return "\n".join(f"{i + 1}. {line}" for i, line in enumerate(_lines(logs)))


def _lines(logs: str) -> list[str]:
    return [line for line in logs.splitlines() if line.strip()]


def _cite_grammar(logs: str) -> str:
    alternatives = " | ".join(f'"{i + 1}"' for i in range(len(_lines(logs))))
    return f'root ::= "NONE"{" | " + alternatives if alternatives else ""}\n'


def _arms(question, source, logs):
    """name -> (prompt, grammar, reading of the answer)."""
    filled = {"question": question, "source": source, "logs": logs}
    return {
        "plain": (
            _PLAIN.format(**filled),
            'root ::= "YES" | "NO"\n',
            lambda raw: raw == "YES",
        ),
        "negated": (
            _NEGATED.format(**filled),
            'root ::= "NO" | "YES"\n',
            lambda raw: raw == "NO",
        ),
        "cite": (
            _CITE.format(question=question, source=source, logs=_numbered(logs)),
            _cite_grammar(logs),
            lambda raw: raw != "NONE",
        ),
        "reason": (
            _REASON.format(**filled),
            _REASON_GRAMMAR,
            lambda raw: raw.endswith("YES"),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        choices=("plain", "negated", "cite", "reason"),
        help="repeatable; default is all four",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="ask each case N times. Temperature is 0 and the answers "
        "still move: llama-server keeps one slot, so what a call reuses "
        "from the KV cache depends on the call before it.",
    )
    args = parser.parse_args()

    from forge.errors import ProviderError
    from forge.llm import call_llm

    arms = args.arm or ["plain", "negated", "cite", "reason"]
    print(f"--- {len(CASES)} fixtures, {len(arms)} arm(s), repeat={args.repeat}\n")

    totals: dict[str, dict[str, int]] = {
        a: {"right": 0, "false-no": 0, "false-yes": 0} for a in arms
    }
    for question, source, logs, answerable in CASES:
        built = _arms(question, source, logs)
        shown = f"{question[:33]:33} | {source[:26]:26} | {'answerable' if answerable else 'no answer '}"
        for arm in arms:
            prompt, grammar, reads_as_yes = built[arm]
            for _ in range(args.repeat):
                started = time.time()
                try:
                    raw = call_llm(prompt, grammar=grammar).strip()
                except ProviderError as e:
                    print(f"{arm:8} PROVIDER FAILED: {e}")
                    return 1
                said_yes = reads_as_yes(raw)
                if said_yes == answerable:
                    verdict, key = "ok  ", "right"
                elif answerable:
                    verdict, key = "FALSE NO ", "false-no"
                else:
                    verdict, key = "FALSE YES", "false-yes"
                totals[arm][key] += 1
                print(
                    f"{arm:8} {verdict:9} ({time.time() - started:4.1f}s) {shown} | {raw[:60]}"
                )
        print()

    print("arm       right  FALSE NO  FALSE YES")
    for arm in arms:
        t = totals[arm]
        print(f"{arm:8} {t['right']:6} {t['false-no']:9} {t['false-yes']:10}")
    print()
    print(
        "A FALSE YES is the status quo: a fluent diagnosis built on logs that\n"
        "do not mention the subject. A FALSE NO refuses a diagnosis that was\n"
        "there. There is deliberately no score and no verdict on the arms --\n"
        "read the columns."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
