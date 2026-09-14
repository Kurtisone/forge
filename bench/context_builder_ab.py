#!/usr/bin/env python3
"""
Does the Context Builder answer better than the log block it would replace?

    PATH=~/.venvs/forge/bin:$PATH PYTHONPATH=src python bench/context_builder_ab.py
    python bench/context_builder_ab.py --only restart --out /tmp/ab.json

WHY THIS EXISTS. forge/kernel/context_builder.py is merged, tested, and
called by nothing. Route A of the wiring plan would put it behind
graphs/sysadmin.py's NO-TARGET path -- the branch that today collects
`journalctl -k` and hands it to _SYNTHESIS_PROMPT. Before any of that,
the question is whether the swap is an improvement or a regression, and
the only honest way to know is to ask the model that will do the work.

WHAT IS ALREADY KNOWN AND NOT RE-ASKED. That the CONTEXT states no
unbacked fact is proven deterministically by tests/test_context_builder*
-- it is a property of the code, not of the model, and no model call can
strengthen it. What is open is downstream: given an honest context, does
this model write a better diagnosis than it writes from kernel logs?

THE TWO ARMS

  logs      graphs.sysadmin._SYNTHESIS_PROMPT, verbatim, imported from
            the module -- today's behaviour, not a reconstruction.
  context   the Context Builder's text, plus the output-shaping tail
            that route A's node would own (see _SHAPING). Same
            "/no_think" at position 0, same GOOD ANSWER example, same
            JSON refusal -- so the arms differ in their EVIDENCE and in
            the framing of that evidence, and in nothing else.

TWO VARIABLES MOVE, AND THAT IS DELIBERATE. The evidence-framing
sentences differ because the Context Builder carries its own reading
rules and _SYNTHESIS_PROMPT carries the ones written for logs. Holding
the framing identical would be a cleaner experiment about a shipping
decision nobody is facing: the question here is "would route A be
better", not "is a fact block intrinsically better than a log block
inside an identical wrapper". Said plainly so no one reads the result
as the second thing.

THE VERDICT IS A HUMAN READ, and that is a finding rather than a
shortcut. bench/sysadmin_verdict.py asked this model whether a log
block answered a question, four ways, over eight fixtures whose answer
was known, and every arm was wrong in the direction its own phrasing
invited. There is no verdict channel at this model size. So this
harness prints both answers in full and counts only what code can
count: whether the answer contains the words the known cause is made
of. That is a PROXY -- an answer can name "RAM" inside a wrong
conclusion -- and it is reported as one.

FIXTURES ARE FIXTURES. The world states below are built as Observations
by hand, not collected: podman is not reachable from the dev sandbox,
and a bench that depended on it could not run where this one runs. The
log blocks are real shapes from this machine.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from forge.harnais.collector import Observation
from forge.harnais.facts import Fact
from forge.kernel.context_builder import ContextBuilder
from forge.kernel.world_model import InMemoryWorldModel

NOW = datetime.now()  # noqa: DTZ005 -- naive local, as everywhere in Forge

# Imported, not repeated: the graph and this harness must ask the
# Context Builder for the same budget, or the bench measures a context
# size nothing ships.
from forge.config import SYSADMIN_CONTEXT_BUDGET_TOKENS as BUDGET_TOKENS

# --------------------------------------------------------------------
# log blocks -- real shapes, from this machine
# --------------------------------------------------------------------

QUIET_KERNEL = """\
[    0.000000] Linux version 6.16.12-valve24.5-1-neptune-616
[    1.884113] systemd[1]: Detected architecture x86-64.
[    2.004221] systemd[1]: Reached target Multi-User System.
[   12.881004] wlan0: associated with 3c:37:86:1f:2a:b0"""

GPU_TIMEOUT = """\
[11204.113] amdgpu 0000:04:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=884, emitted seq=886
[11204.115] amdgpu 0000:04:00.0: amdgpu: GPU reset begin!
[11206.884] amdgpu 0000:04:00.0: amdgpu: GPU reset succeeded, trying to resume
[11207.001] [drm] PCIE GART of 512M enabled"""


# --------------------------------------------------------------------
# world states
# --------------------------------------------------------------------


def _containers(uptimes: dict[str, int]) -> Observation:
    facts = [Fact("container", "running_count", len(uptimes), None, NOW, "containers")]
    for name, up in uptimes.items():
        human = f"Up {up // 3600} hours" if up >= 3600 else f"Up {up // 60} minutes"
        facts.append(
            Fact("container", f"{name}.status", human, None, NOW, "containers")
        )
        facts.append(Fact("container", f"{name}.uptime_s", up, "s", NOW, "containers"))
    return Observation.of("containers", ("container",), facts, NOW)


def _cpu_ram(used_pct: float, load: float) -> Observation:
    total = 15160368
    facts = [
        Fact("ram", "total_kb", total, "kB", NOW, "cpu_ram"),
        Fact(
            "ram",
            "available_kb",
            int(total * (1 - used_pct / 100)),
            "kB",
            NOW,
            "cpu_ram",
        ),
        Fact("ram", "used_pct", used_pct, "%", NOW, "cpu_ram"),
        Fact("cpu", "load_1m", load, None, NOW, "cpu_ram"),
        Fact("cpu", "load_5m", round(load * 0.8, 2), None, NOW, "cpu_ram"),
        Fact("cpu", "load_15m", round(load * 0.6, 2), None, NOW, "cpu_ram"),
    ]
    return Observation.of("cpu_ram", ("cpu", "ram"), facts, NOW)


def _logs(block: str, source: str = "journalctl -k") -> Observation:
    """
    The logs domain, from `source`.

    `source` is not always the kernel. Route C's question is whether a
    VALIDATED per-unit collection can be recorded as a Fact like any
    other and rendered alongside the machine's state -- which is what
    collectors/logs.py said would have to happen before per-unit logs
    could exist at all ("per-unit logs arrive with the validator that
    makes them safe, not before"). Here it is a fixture; in the graph
    it would be _collect_node's own output, already checked against
    that run's discovery.
    """
    lines = block.splitlines()
    facts = [
        Fact("logs", f"{source}.lines", len(lines), "lines", NOW, "logs"),
        Fact("logs", f"{source}.tail", block, None, NOW, "logs"),
    ]
    return Observation.of("logs", ("logs",), facts, NOW)


def _world(*observations) -> InMemoryWorldModel:
    world = InMemoryWorldModel()
    for observation in observations:
        world.record_observation(observation)
    return world


ALL_UP = {
    "forge": 10800,
    "forge-llm": 10800,
    "forge-embedding": 10800,
    "searxng": 10800,
}
LLM_JUST_RESTARTED = {**ALL_UP, "forge-llm": 247}


# --------------------------------------------------------------------
# the fixtures
# --------------------------------------------------------------------
#
# `cause` is the set of words a correct answer is made of. It is a
# PROXY and is reported as one -- see the module docstring.

SEARXNG_OOM = """\
2026-09-14 09:11:57,003 WARNING:searx.engines: engine timeout: google
2026-09-14 09:11:58,441 INFO:searx.webapp: shutting down
[11223.884] Out of memory: Killed process 4412 (searxng-run) total-vm:2118344kB
2026-09-14 09:12:03,114 INFO:searx.webapp: starting webserver on http://0.0.0.0:8080/
2026-09-14 09:12:04,882 INFO:searx.engines: 92 engines loaded"""

LLM_QUIET = """\
llama_context: n_ctx_per_seq (16384) < n_ctx_train (32768) -- the full capacity will not be used
llama_context: CPU output buffer size = 0.58 MiB
main: server is listening on http://0.0.0.0:8080 - starting the main loop
srv update_slots: all slots are idle"""

PODMAN_DOWN = (
    "[error] podman exited 125: unable to connect to Podman socket: "
    "dial unix /run/forge-podman-ro-proxy/sock: connect: no such file or directory"
)

FIXTURES = [
    {
        "name": "restart",
        "question": "Forge a été très lent il y a dix minutes, pourquoi ?",
        "logs": QUIET_KERNEL,
        "world": lambda: _world(
            _containers(LLM_JUST_RESTARTED), _cpu_ram(41.2, 0.9), _logs(QUIET_KERNEL)
        ),
        "cause": ["forge-llm", "redémarr", "relanc", "247", "4 minutes"],
        "known": "forge-llm restarted 247 s ago while everything else has 3 h of "
        "uptime. The kernel log cannot show this, so the logs arm cannot win it.",
    },
    {
        "name": "ram",
        "question": "Ma machine rame, qu'est-ce qui se passe ?",
        "logs": QUIET_KERNEL,
        "world": lambda: _world(
            _containers(ALL_UP), _cpu_ram(94.3, 7.8), _logs(QUIET_KERNEL)
        ),
        "cause": ["mémoire", "ram", "94", "charge", "load"],
        "known": "94 % of memory used and a load average of 7.8. Quiet kernel log, "
        "so again invisible to the logs arm.",
    },
    {
        "name": "gpu",
        "question": "J'ai eu un freeze graphique tout à l'heure, qu'est-ce qui s'est passé ?",
        "logs": GPU_TIMEOUT,
        "world": lambda: _world(
            _containers(ALL_UP), _cpu_ram(38.0, 1.1), _logs(GPU_TIMEOUT)
        ),
        "cause": ["gpu", "amdgpu", "reset", "timeout"],
        "known": "THE CONTROL ARM. The answer is in the kernel log, which both arms "
        "carry. The context arm must not do WORSE here -- if the facts around the "
        "log block distract from it, this is where that shows.",
    },
    # --- route C: the NAMED-target path ----------------------------------
    #
    # These two exist to test a claim made without measuring it -- that
    # route C is blocked by the `blind` result. It may not be. On the
    # targeted path, `synthesize` is reached ONLY when the target was
    # found in this run's own discovery AND its logs were collected AND
    # they are not empty, so the SUBJECT is observed by construction and
    # `blind`'s precondition cannot occur there.
    #
    # What can still occur is a DIFFERENT domain being dark while the
    # question names something in another one. That is the real question,
    # and it is what these ask.
    {
        "name": "unit_ok",
        "question": "Pourquoi searxng redémarre en boucle ?",
        "source": "journalctl -u searxng.service",
        "logs": SEARXNG_OOM,
        "world": lambda: _world(
            _containers(ALL_UP),
            _cpu_ram(94.3, 7.8),
            _logs(SEARXNG_OOM, "journalctl -u searxng.service"),
        ),
        "cause": ["mémoire", "oom", "out of memory", "tué", "killed", "94"],
        "known": "The unit's own logs carry an OOM kill, and memory is at 94 %. "
        "Both arms have the log; only the context arm can corroborate it with the "
        "memory figure. The value case for route C.",
    },
    {
        "name": "unit_blind",
        "question": "Pourquoi searxng redémarre en boucle ?",
        "source": "journalctl -u searxng.service",
        "logs": SEARXNG_OOM,
        "world": lambda: _world(
            Observation.failure("containers", ("container",), PODMAN_DOWN, NOW),
            _cpu_ram(41.2, 0.9),
            _logs(SEARXNG_OOM, "journalctl -u searxng.service"),
        ),
        "cause": ["mémoire", "oom", "out of memory", "tué", "killed"],
        "risk": ["podman", "socket", "proxy", "/run/forge"],
        "known": "THE RISK CASE, and the one holding route C back. The subject IS "
        "observed -- its logs are right there and they say OOM -- but ANOTHER "
        "domain is dark, with a podman error in the context. If the model blames "
        "the podman socket for searxng restarting, route C stays blocked. If it "
        "reads the log it was given, the `blind` result does not transfer and the "
        "block was my assumption, not a measurement.",
    },
    {
        "name": "unit_quiet",
        "question": "Pourquoi forge-llm répond si lentement ?",
        "source": "journalctl -u forge-llm",
        "logs": LLM_QUIET,
        "world": lambda: _world(
            _containers(ALL_UP),
            _cpu_ram(94.3, 7.8),
            _logs(LLM_QUIET, "journalctl -u forge-llm"),
        ),
        "cause": ["mémoire", "ram", "94", "charge", "load", "swap"],
        "known": "THE VALUE CASE for route C, and the `ram` fixture transposed "
        "onto the targeted path. The unit's own logs are healthy startup notices "
        "-- they cannot explain slowness -- while memory sits at 94 % and the load "
        "average at 7.8. The logs arm has only the quiet log and can at best say "
        "so; the context arm has the machine. If it does not win here, route C "
        "buys uniformity and nothing else.",
    },
    {
        "name": "blind",
        "question": "Pourquoi forge-llm plante ?",
        "logs": QUIET_KERNEL,
        "world": lambda: _world(
            Observation.failure(
                "containers",
                ("container",),
                "[error] podman exited 125: unable to connect to Podman socket: "
                "dial unix /run/forge-podman-ro-proxy/sock: connect: no such file "
                "or directory",
                NOW,
            ),
            _cpu_ram(41.2, 0.9),
            _logs(QUIET_KERNEL),
        ),
        "cause": ["pas pu", "impossible", "proxy", "podman", "socket", "observ"],
        "known": "Run #83fc443e, as it stood on this machine from 2026-09-11 to "
        "09-14. The container state cannot be read. The correct answer is to say so "
        "and name the broken command -- NOT to diagnose the container from a kernel "
        "log that never mentions it.",
    },
]


# --------------------------------------------------------------------
# the two prompts
# --------------------------------------------------------------------


#: The output-shaping half, imported from the graph rather than copied.
#:
#: It lived here as a literal while route A did not exist -- the same
#: position bench/sysadmin_verdict.py writes from, a harness testing an
#: idea before the code. Route A shipped, so the copy became the thing
#: CLAUDE.md forbids: a bench measuring a prompt the graph does not
#: send. Two places stating the same prompt, one of them edited, and
#: this harness would keep reporting on a prompt nobody runs.
def logs_prompt(fixture) -> str:
    """Today's behaviour, from the real module -- not a reconstruction."""
    from forge.config import SYSADMIN_LOG_CHARS_BUDGET
    from forge.context_info import today_line
    from forge.graphs.sysadmin import _SYNTHESIS_PROMPT, _truncate_log_block

    return _SYNTHESIS_PROMPT.format(
        today_line=today_line(),
        question=fixture["question"],
        source=fixture.get("source", "journalctl -k"),
        log_block=_truncate_log_block(fixture["logs"], SYSADMIN_LOG_CHARS_BUDGET),
        running_fact="",
    )


def context_prompt(fixture) -> str:
    context = ContextBuilder(fixture["world"]()).build_for(
        fixture["question"], BUDGET_TOKENS
    )
    from forge.graphs.sysadmin import _CONTEXT_SHAPING

    return "/no_think\n" + context + "\n" + _CONTEXT_SHAPING


def terse_prompt(fixture) -> str:
    """
    EXPLORATION ARM, added after the first pass, and not a wording fix.

    The `context` arm turned an observation failure into a cause:
    "forge-llm plante CAR le socket de Podman n'existe pas". The context
    said the instrument was broken; the model read it as the diagnosis.
    Stable over three runs at temperature 0.

    _RULES already says "never a diagnosis" in as many words, so the
    fix cannot be to say it better -- that is the shape this codebase
    has watched fail thirteen times. It has to be something the model
    cannot use: if the podman error text is not in the context, no
    causal story can be built out of it.

    The error does not disappear, it changes channel. Naming the broken
    command is the single most useful thing a reader gets -- sysadmin's
    _collect_failed_node exists to say so -- and that node writes it in
    CODE, never through the model. Same split here: the model is told
    the domain was not observed, the caller reports why.

    Implemented as a transformation of the builder's own marker lines
    rather than a rewritten prompt, so this measures the real context
    minus one thing.

    VERDICT 2026-09-14: DEAD, and worse than what it replaced. Do not
    retry it. Stable over two runs:

        plante car la charge CPU est élevée (0.9 sur 1 minute) et la
        mémoire disponible est faible (41.2 % utilisé)

    A load average of 0.9 is not high and 41 % of memory used is not
    low. Starved of the podman error, the model reached for the next
    nearest facts and declared HEALTHY ones the cause -- which is worse
    than the `context` arm, that at least pointed at something really
    broken.

    What it establishes is bigger than the arm: asked why a named thing
    crashes, this model returns a cause no matter what the context
    holds. Removing a candidate does not produce a refusal, it produces
    the next candidate. So no arrangement of FACTS fixes this case, and
    the only thing that can is not calling the model at all --
    precisely the conclusion graphs/sysadmin.py already reached with
    target_missed, collect_failed and nothing_collected, three nodes
    written entirely in code because a model handed evidence about
    something else answered fluently anyway.
    """
    from forge.kernel.context_builder import UNOBSERVED_PREFIX

    full = context_prompt(fixture)
    out = []
    for line in full.splitlines():
        if line.startswith(UNOBSERVED_PREFIX):
            domain = line[len(UNOBSERVED_PREFIX) :].split(":", 1)[0]
            out.append(f"{UNOBSERVED_PREFIX}{domain}: not observed")
        else:
            out.append(line)
    return "\n".join(out)


ARMS = {"logs": logs_prompt, "context": context_prompt, "terse": terse_prompt}


# --------------------------------------------------------------------
# running
# --------------------------------------------------------------------


def _erase_slot() -> None:
    """Same cold-start courtesy as bench/router_ab.py. Best effort."""
    import requests

    from forge.config import LLAMA_CPP_ID_SLOT, LLAMA_CPP_URL

    try:
        requests.post(
            f"{LLAMA_CPP_URL}/slots/{LLAMA_CPP_ID_SLOT}?action=erase", timeout=10
        )
    except Exception as e:  # noqa: BLE001 - never fatal
        print(f"  (slot erase failed: {e})")


def run_one(fixture, arm) -> dict:
    from forge.graphs.sysadmin import _clean_diagnosis_response
    from forge.lang import line_for
    from forge.llm import call_llm

    prompt = ARMS[arm](fixture)
    started = time.monotonic()
    try:
        raw = call_llm(prompt + line_for(fixture["question"]))
        answer = _clean_diagnosis_response(raw)
        error = None
    except Exception as e:  # noqa: BLE001 - record, never abort the run
        answer, error = "", str(e)
    wall_ms = int((time.monotonic() - started) * 1000)

    lowered = answer.lower()
    hits = [w for w in fixture["cause"] if w.lower() in lowered]
    # Words that would mean the model blamed something it was told it
    # could not SEE. Separate from a missing cause: naming the wrong
    # culprit and failing to name the right one are different failures,
    # and only one of them is the one route C is held back by.
    risk = [w for w in fixture.get("risk", []) if w.lower() in lowered]
    # A dead server and a wrong answer are not the same result, and the
    # first run of this harness printed both as "MISS". llama-server
    # fell over mid-pass on 2026-09-14 (RemoteDisconnected, 1091 ms)
    # and `restart: unless-stopped` revived it, so the rows either side
    # of it came from two different processes -- which, at temperature
    # 0, is exactly the boundary across which this model's answers are
    # known to shift. bench/research_ceiling.py already learned to say
    # DOWN; this one was not born knowing it either.
    return {
        "fixture": fixture["name"],
        "arm": arm,
        "prompt_chars": len(prompt),
        "wall_ms": wall_ms,
        "answer": answer,
        "answer_chars": len(answer),
        "cause_words_hit": hits,
        "risk_words_hit": risk,
        "error": error,
    }


def _write(args, rows) -> None:
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="run a single fixture by name")
    p.add_argument("--out", help="write the full result as JSON")
    args = p.parse_args()

    # Without this the router grammar falls back to a reduced tool set
    # -- 677 characters instead of the 1005 the deployed ENABLED_TOOLS
    # produces -- and the run measures a router that exists nowhere.
    from forge.tools import registry

    registry.load_tools()

    fixtures = [f for f in FIXTURES if not args.only or f["name"] == args.only]
    if not fixtures:
        raise SystemExit(f"no such fixture: {args.only!r}")

    _erase_slot()
    rows = []
    consecutive_down = 0
    for fixture in fixtures:
        print(f"\n{'=' * 70}\n{fixture['name']}: {fixture['question']}")
        print(f"known: {fixture['known']}")
        for arm in ARMS:
            row = run_one(fixture, arm)
            rows.append(row)
            consecutive_down = consecutive_down + 1 if row["error"] else 0
            if consecutive_down >= 3:
                print(
                    "\nSTOPPING: three calls in a row failed to reach "
                    "llama-server. The rows above may straddle a restart; "
                    "read them as two runs, not one.",
                    flush=True,
                )
                _write(args, rows)
                return
            # Line by line, flushed: a long run redirected to a file
            # must not look identical whether it is advancing or wedged.
            print(
                f"\n--- {arm} ({row['prompt_chars']} chars in, "
                f"{row['wall_ms']} ms, cause: "
                f"{row['cause_words_hit'] or 'NONE'}"
                + (f", RISK: {row['risk_words_hit']}" if row["risk_words_hit"] else "")
                + ") ---",
                flush=True,
            )
            print(row["error"] or row["answer"], flush=True)

    print(f"\n{'=' * 70}\nsummary (cause-word hits are a PROXY, read the answers)")
    down = 0
    for row in rows:
        if row["error"]:
            mark, down = "DOWN", down + 1
        else:
            mark = "hit " if row["cause_words_hit"] else "MISS"
        risk = "  <-- RISK WORDS" if row["risk_words_hit"] else ""
        print(
            f"  {mark} {row['fixture']:>10} / {row['arm']:<8} "
            f"{row['wall_ms']:>6} ms{risk}"
        )
    if down:
        print(
            f"\n  {down} call(s) never reached llama-server. Those are not "
            "verdicts, and any row after one may come from a different "
            "server process than the rows before it."
        )

    _write(args, rows)


if __name__ == "__main__":
    main()
