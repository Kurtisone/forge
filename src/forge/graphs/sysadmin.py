"""
Forge sysadmin graph: discover -> collect -> synthesize.

Same reasoning as graphs/research.py: a deterministic fixed sequence,
never a router-driven multi-step chain. The router makes exactly ONE
decision (call "sysadmin"), the graph itself decides what to run next
-- no mid-flow judgment call is ever handed back to the model.

Security model (read-only, always):
  - discover_node runs two fixed, parameter-free commands
    (a busctl ListUnits call, podman ps) -- nothing here can be
    influenced by user input, so there is no injection surface at
    this step.
  - collect_node only ever runs a command built from a fixed template
    (see _COLLECT_TEMPLATES) whose one variable slot (a unit or
    container name) must appear verbatim in the list discover_node
    just returned. A name that isn't in that list is refused before
    any subprocess is started -- this is the entire threat model:
    nothing reaches subprocess.run() that wasn't independently
    confirmed to already exist on the system, and no shell=True is
    ever used, so there is no string-interpolation-into-a-shell
    surface either.
  - No mutation command exists in this module, at all. Restarting or
    stopping anything is explicitly out of scope for v3.11 -- this
    graph only ever reads logs and proposes; see the module docstring
    in tools/sysadmin.py for the human-in-the-loop reasoning, the
    same one already applied to tools/git.py staying read-only.

Nodes:
  discover_node    -- lists active systemd units and podman containers
  collect_node      -- fetches logs for one validated target (or
                        kernel logs as a fallback when no target was
                        named or found)
  synthesize_node   -- single LLM call producing a diagnosis and a
                        proposed fix, never an executed action

Edges:
  discover_node  -> collect_node     (always)
  collect_node   -> synthesize_node  (always -- a failed individual
                                       collection still gets surfaced
                                       to the model as "[error] ...",
                                       it isn't fatal to the run)

Usage (Python):
  from forge.graphs.sysadmin import run
  print(run(target_hint="searxng", question="pourquoi ça redémarre ?"))
"""

import json
import subprocess

from forge import lang, prose_grammar, subtrace
from forge.config import (
    ENFORCE_ANSWER_LANGUAGE,
    SYSADMIN_COLLECT_TIMEOUT,
    SYSADMIN_DBUS_ADDRESS,
    SYSADMIN_DISCOVERY_TIMEOUT,
    SYSADMIN_JOURNAL_DIR,
    SYSADMIN_LOG_CHARS_BUDGET,
    SYSADMIN_MAX_LOG_LINES,
    SYSADMIN_PODMAN_URL,
)
from forge.context_info import today_line
from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.text_cleaning import strip_think_blocks, try_unwrap_router_json
from forge.types import AgentState

_MAX_SYNTHESIS_OUTPUT_CHARS = 4000

# Same reasoning as graphs/review.py and graphs/research.py: this
# model needs the exact shape it must NOT produce shown explicitly.
# _PROMPT_LEAK_MARKERS catches the model echoing the INSTRUCTION
# markers themselves (rare). _EXAMPLE_LEAK_FRAGMENT catches the
# actually-observed failure mode: the model silently copying the
# GOOD ANSWER *content* verbatim, without ever echoing the marker
# word -- caught in production on 2026-08-11, where a question about
# a completely different container ("forge-llm") got this exact
# fabricated searxng/port-8888 text back as if it were a real
# diagnosis. A distinctive fragment of the example is enough: an
# unrelated real diagnosis coincidentally using this exact phrase is
# effectively impossible.
_PROMPT_LEAK_MARKERS = [
    "Respond in plain text",
    "GOOD ANSWER:",
    "NEVER DO THIS",
]
_EXAMPLE_LEAK_FRAGMENTS = [
    "port 8888 est déjà occupé",  # previous example, kept as a permanent safety net
    "exemple-service.service",
    "manquant.conf",
]


# Fixed, parameter-free discovery commands. Functions, not static
# lists: SYSADMIN_DBUS_ADDRESS/SYSADMIN_PODMAN_URL let these target a
# filtered proxy instead of the raw host bus/socket -- see config.py's
# comment above these three env vars, and deploy/README.md for the
# proxies themselves. Empty (default, incl. every test in this file)
# means "unchanged": no proxy configured, no extra flag added, exact
# same command as before this was made configurable.
def _DISCOVER_UNITS_CMD() -> list[str]:
    # `systemctl list-units` was the original approach but had to be
    # abandoned: confirmed in production (SYSTEMD_LOG_LEVEL=debug)
    # that systemctl hardcodes a connection attempt at
    # /run/systemd/private first -- a systemd-specific shortcut
    # protocol, NOT standard D-Bus -- and never falls back to
    # DBUS_SYSTEM_BUS_ADDRESS (or any other address) if that exact
    # path is unavailable, which it always is inside a container whose
    # PID 1 isn't systemd. `busctl` has no such quirk: it speaks
    # standard D-Bus and honors --address correctly, confirmed
    # repeatedly against the same filtered proxy that systemctl
    # refused to use. --json=short gives a real parseable structure
    # (see _parse_busctl_units) instead of the columnar text
    # `systemctl list-units` produces.
    cmd = ["busctl", "--json=short"]
    if SYSADMIN_DBUS_ADDRESS:
        cmd.append(f"--address={SYSADMIN_DBUS_ADDRESS}")
    return cmd + [
        "call",
        "org.freedesktop.systemd1",
        "/org/freedesktop/systemd1",
        "org.freedesktop.systemd1.Manager",
        "ListUnits",
    ]


def _parse_busctl_units(raw: str) -> list[str]:
    """ListUnits' D-Bus signature is a(ssssssouso) -- an array of
    10-field tuples (name, description, load_state, active_state,
    sub_state, following, unit_path, job_id, job_type, job_path).
    `busctl --json=short` wraps that as {"type": "...", "data": [rows]}
    where `data[0]` is the array of tuples and each tuple's index 0 is
    the unit name -- verified against real output in production
    before writing this, not guessed from the D-Bus spec alone."""
    parsed = json.loads(raw)
    rows = parsed["data"][0]
    return [row[0] for row in rows if row]


def _DISCOVER_CONTAINERS_CMD() -> list[str]:
    base = ["podman"]
    if SYSADMIN_PODMAN_URL:
        base += ["--url", SYSADMIN_PODMAN_URL]
    return base + ["ps", "--format", "{{.Names}}"]


def _collect_cmd(kind: str, name: str) -> list[str]:
    """Build a collection command. {name} is substituted only after
    collect_node has verified it against discover_node's own output --
    see collect_node's docstring. `kind` selects journalctl-by-unit,
    podman-logs, or journalctl-kernel; the journal dir / podman URL
    flags are added only when the matching proxy is configured.

    The name is passed as `--unit=<name>` and after a `--` separator
    respectively (audit M-4), never as a bare argument following a
    short flag. `-u foo` and `foo` are both positions where a value
    starting with `-` is read as an option instead: `-u --output=cat`
    or a container literally named `--help` would be interpreted by
    journalctl/podman rather than treated as a name. The attached form
    and the end-of-options marker remove that reading entirely.

    This is the second lock on a door that collect_node already
    bolted: a name only reaches here if it appeared verbatim in
    discovery output, and unit/container names don't normally start
    with a dash. What it defends is the case where discovery output is
    no longer trustworthy -- a hostile container name, or a proxy
    returning something the host didn't say -- which is exactly the
    assumption the validation rests on and therefore the one worth not
    resting the whole thing on.
    """
    if kind == "unit":
        cmd = ["journalctl"]
        if SYSADMIN_JOURNAL_DIR:
            cmd += ["-D", SYSADMIN_JOURNAL_DIR]
        return cmd + [f"--unit={name}", "--no-pager", "-n", str(SYSADMIN_MAX_LOG_LINES)]
    if kind == "container":
        cmd = ["podman"]
        if SYSADMIN_PODMAN_URL:
            cmd += ["--url", SYSADMIN_PODMAN_URL]
        return cmd + ["logs", "--tail", str(SYSADMIN_MAX_LOG_LINES), "--", name]
    if kind == "kernel":
        cmd = ["journalctl"]
        if SYSADMIN_JOURNAL_DIR:
            cmd += ["-D", SYSADMIN_JOURNAL_DIR]
        return cmd + ["-k", "--no-pager", "-n", str(SYSADMIN_MAX_LOG_LINES)]
    raise ValueError(f"unknown collect kind: {kind!r}")


def _subprocess_env() -> dict[str, str]:
    """Same minimal-env posture as tools/shell.py: no host env
    variables reach the subprocess except what's explicitly listed.
    DBUS_SYSTEM_BUS_ADDRESS is added only when SYSADMIN_DBUS_ADDRESS
    is configured, pointing busctl at the filtered proxy socket
    from deploy/forge-dbus-proxy.sh -- never the real system bus."""
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "TERM": "dumb"}
    if SYSADMIN_DBUS_ADDRESS:
        env["DBUS_SYSTEM_BUS_ADDRESS"] = SYSADMIN_DBUS_ADDRESS
    return env


# The "/no_think" prefix below is NOT dead, however dead it looks.
# Qwen3.5 dropped the /think soft switch, and the router GBNF grammar
# (applied to every call, not just routing) already makes a reasoning
# block impossible -- so on paper the token buys nothing. Measured on
# 2026-08-16 with bench/no_think_ab.py, removing it made this model
# return the GOOD ANSWER example below instead of a real answer, twice,
# deterministically. Whatever it does at position 0 is not what its
# name says. Run that harness before touching it.
_SYNTHESIS_PROMPT = """/no_think
{today_line}
You are diagnosing a system/service problem using the logs gathered
for you below. Write a clear, natural diagnosis in plain text --
explain what is likely happening and propose a concrete fix. Write in
the same language as the question. Never claim you have executed or
will execute anything: you only read logs and propose, a human always
applies any fix by hand.

Question: {question}

--- collected logs ({source}) ---
{log_block}
--- end of collected logs ---

Respond in plain text ONLY. Do NOT wrap your answer in JSON, and do
NOT return a {{"tool":...,"content":...}} object -- that format is
for a different system (a routing decision) and never applies here.

GOOD ANSWER (this is only an example of FORM AND TONE -- these exact
names, files and details are fictional placeholders, not real logs;
copying any of them into your own answer is always wrong, no matter
what the actual logs above say): Le service exemple-service.service
échoue au démarrage car la configuration référence un fichier
introuvable (/etc/exemple/manquant.conf). Je te propose de vérifier
que ce fichier existe et, si besoin, de le recréer avant de relancer
le service.
NEVER DO THIS: {{"tool":"chat","content":"..."}}

Now write your own diagnosis using ONLY what actually appears in the
collected logs above -- the words "exemple-service" and
"manquant.conf" must never appear in your answer, they belong to the
placeholder example only, never to a real one. Same plain format as
GOOD ANSWER, not the NEVER DO THIS shape. Be concise.
"""


def _run_fixed(cmd: list[str], timeout: int) -> str:
    """Run a command whose every element is either a fixed literal or
    a name already verified against discover_node's own output.
    Never shell=True, never a hand-built string -- same posture as
    tools/shell.py's allowlisted subprocess.run(parts, ...). Uses
    _subprocess_env() so DBUS_SYSTEM_BUS_ADDRESS (when configured)
    points busctl at the filtered proxy, not the host bus."""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_subprocess_env(),
        )
    except FileNotFoundError:
        return f"[error] executable not found: {cmd[0]!r}"
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except OSError as e:
        return f"[error] OS error: {e}"

    output = (result.stdout or result.stderr or "").strip()
    lines = output.splitlines()[:SYSADMIN_MAX_LOG_LINES]
    joined = "\n".join(lines) if lines else "[no output]"

    if result.returncode != 0:
        # The command RAN (no Python-level exception above) but the
        # target itself failed -- e.g. busctl unable to reach the bus,
        # podman unable to reach its socket. This must carry the same
        # "[error]" prefix as the exception-based cases above:
        # without it, a real production case slipped straight through
        # as if it were valid data. The case that taught this was
        # systemctl, back when discovery still used it: its two-line
        # failure message ("System has not been booted with
        # systemd...\nFailed to connect to bus...") got parsed as two
        # fake unit names ("System", "Failed") by _discover_node, and
        # podman's connection-refused text got parsed as a fake
        # container name the same way. Caught in production on
        # 2026-08-11. The systemctl path is gone; the failure mode it
        # exposed is not, which is why the guard stays.
        return f"[error] {cmd[0]} exited {result.returncode}: {joined}"

    return joined


def _discover_node(state: AgentState) -> AgentState:
    units_raw = _run_fixed(_DISCOVER_UNITS_CMD(), SYSADMIN_DISCOVERY_TIMEOUT)
    if units_raw.startswith("[error]"):
        # e.g. "executable not found: 'busctl'" -- systemd tooling
        # isn't necessarily present inside Forge's own container image.
        # Must not be parsed as a fake unit named "[error]".
        log.warning("sysadmin: unit discovery failed: %s", units_raw)
        units: list[str] = []
        state.context["discover_units_error"] = units_raw
    else:
        try:
            units = _parse_busctl_units(units_raw)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            log.warning("sysadmin: failed to parse busctl ListUnits output: %s", e)
            units = []
            state.context["discover_units_error"] = (
                f"[error] failed to parse busctl output: {e}"
            )

    containers_raw = _run_fixed(_DISCOVER_CONTAINERS_CMD(), SYSADMIN_DISCOVERY_TIMEOUT)
    if containers_raw.startswith("[error]"):
        log.warning("sysadmin: container discovery failed: %s", containers_raw)
        containers: list[str] = []
        state.context["discover_containers_error"] = containers_raw
    else:
        containers = [
            line.strip() for line in containers_raw.splitlines() if line.strip()
        ]

    state.context["units"] = units
    state.context["containers"] = containers
    log.event("sysadmin.discover", units=len(units), containers=len(containers))
    return state


def _collect_node(state: AgentState) -> AgentState:
    """Picks exactly one collection target. target_hint must appear
    verbatim in discover_node's own units/containers list to be used
    at all -- an unrecognised hint silently falls back to kernel logs
    rather than being passed to a command, so there is no path from
    unvalidated router content to subprocess.run()."""
    target_hint = state.context.get("target_hint")
    units = state.context["units"]
    containers = state.context["containers"]

    if target_hint and target_hint in units:
        cmd = _collect_cmd("unit", target_hint)
        source = f"journalctl -u {target_hint}"
    elif target_hint and target_hint in containers:
        cmd = _collect_cmd("container", target_hint)
        source = f"podman logs {target_hint}"
    else:
        if target_hint:
            log.warning(
                "sysadmin: target_hint %r not found in discovery, falling back to kernel logs",
                target_hint,
            )
        cmd = _collect_cmd("kernel", "")
        source = "journalctl -k"

    state.context["log_source"] = source
    state.context["collected_logs"] = _run_fixed(cmd, SYSADMIN_COLLECT_TIMEOUT)
    log.event("sysadmin.collect", source=source)
    return state


def _clean_diagnosis_response(raw: str) -> str:
    """Same reasoning as review._clean_review_response and
    research._clean_synthesis_response (see forge/text_cleaning.py):
    this prompt asks for plain text, so the shared conditional-unwrap
    cleaner is reused here too, specifically to avoid the duplication
    drift already hit once between review and research in v3.10."""
    cleaned = strip_think_blocks(raw)

    unwrapped = try_unwrap_router_json(cleaned, source="sysadmin")
    if unwrapped is not None:
        cleaned = unwrapped

    if any(marker in cleaned for marker in _PROMPT_LEAK_MARKERS):
        log.warning("sysadmin: model echoed prompt instructions instead of answering")
        return "[error] Le modèle n'a pas généré de réponse exploitable. Réessayez."

    if any(fragment in cleaned for fragment in _EXAMPLE_LEAK_FRAGMENTS):
        log.warning(
            "sysadmin: model copied the GOOD ANSWER example verbatim "
            "instead of diagnosing the actual logs"
        )
        return "[error] Le modèle a recopié un exemple au lieu de générer un vrai diagnostic. Réessayez."

    if not cleaned:
        return "[error] Le modèle n'a pas généré de réponse. Réessayez."

    if len(cleaned) > _MAX_SYNTHESIS_OUTPUT_CHARS:
        cleaned = cleaned[:_MAX_SYNTHESIS_OUTPUT_CHARS].rstrip() + "…"

    return cleaned


def _truncate_log_block(log_block: str, budget: int) -> str:
    """Hard character cap independent of line count -- SYSADMIN_MAX_LOG_LINES
    alone isn't a safe context guarantee (line length varies a lot
    between journalctl/podman output). Keeps the END of the block:
    _run_fixed already returns at most the tail N lines via -n/--tail,
    so the most recent -- most relevant -- events are at the end."""
    if len(log_block) <= budget:
        return log_block
    return "…[troncated, showing the most recent output]…\n" + log_block[-budget:]


def _synthesize_node(state: AgentState) -> AgentState:
    question = (
        state.context.get("question")
        or "Diagnostique le problème et propose une solution."
    )
    source = state.context["log_source"]
    log_block = _truncate_log_block(
        state.context["collected_logs"], SYSADMIN_LOG_CHARS_BUDGET
    )

    prompt = _SYNTHESIS_PROMPT.format(
        today_line=today_line(),
        question=question,
        source=source,
        log_block=log_block,
    )

    # Language named in LAST position, and only when forge.lang is
    # sure -- same treatment recall got in the v3.12 dettes batch, for
    # the same reason: this prompt body is English prose, and it pulls
    # a French answer toward English all on its own. Appended rather
    # than templated in, so "last" cannot drift as the template grows.
    language_line = lang.line_for(question)

    log.event("sysadmin.llm_call", source=source, prompt_chars=len(prompt))
    try:
        # PROSE, not the router grammar -- see forge/prose_grammar.py.
        # This graph has logged "model wrapped a substantive answer in
        # router-style JSON" since v3.11; that was the cause.
        raw = call_llm(prompt + language_line, grammar=prose_grammar.PROSE)
        log.event("sysadmin.raw_output", raw=raw)
        answer = _clean_diagnosis_response(raw)
        # The deterministic half. Naming the language in the prompt is
        # still a wording fix, and wording fixes have lost seven times
        # on this codebase. The retry re-sends the same prompt with a
        # different final line, so the KV prefix survives and only the
        # tail is recomputed.
        answer = lang.enforce(
            question,
            answer,
            retry=lambda line: _clean_diagnosis_response(
                call_llm(prompt + line, grammar=prose_grammar.PROSE)
            ),
            enabled=ENFORCE_ANSWER_LANGUAGE,
        )
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    state.final_output = answer
    state.final_tool = "sysadmin"
    log.event("sysadmin.done", chars=len(state.final_output))
    return state


def build() -> Graph:
    g = Graph("sysadmin", max_steps=6)
    g.add_node("discover", _discover_node)
    g.add_node("collect", _collect_node)
    g.add_node("synthesize", _synthesize_node)

    g.add_edge("discover", "collect")
    g.add_edge("collect", "synthesize")

    return g


def _format_discovered_list(names: list[str], max_shown: int = 8) -> str:
    """Names, not just a count -- capped so a host with dozens of
    active units doesn't produce an unreadable wall of text in the
    UI's expanded step detail."""
    if not names:
        return "aucun"
    shown = ", ".join(names[:max_shown])
    if len(names) > max_shown:
        shown += f", … (+{len(names) - max_shown})"
    return shown


def _to_sub_steps(state: AgentState) -> list[dict]:
    """Turn this run's internal graph trace (already recorded node by
    node by graph.py's Node.execute) into small, human-readable steps
    for the UI -- see forge/subtrace.py's docstring for why this is a
    separate publish rather than a widened tool contract."""
    units = state.context.get("units", [])
    containers = state.context.get("containers", [])
    units_error = state.context.get("discover_units_error")
    containers_error = state.context.get("discover_containers_error")
    collected_logs = state.context.get("collected_logs", "")
    collect_error = collected_logs if collected_logs.startswith("[error]") else None

    def discover_detail() -> str:
        services_part = (
            f"services : erreur ({units_error})"
            if units_error
            else f"services : {_format_discovered_list(units)}"
        )
        containers_part = (
            f"containers : erreur ({containers_error})"
            if containers_error
            else f"containers : {_format_discovered_list(containers)}"
        )
        return f"{services_part} | {containers_part}"

    def collect_detail() -> str:
        source = state.context.get("log_source", "?")
        if collect_error:
            return f"source : {source} | erreur : {collect_error}"
        return f"source : {source}"

    details = {
        "discover": discover_detail,
        "collect": collect_detail,
        "synthesize": lambda: (
            f"diagnostic généré ({len(state.final_output or '')} caractères)"
        ),
    }
    return [
        {
            "label": ts.decision_tool,
            "detail": details.get(ts.decision_tool, lambda: "")(),
            # A discover/collect step with a real subprocess error is
            # flagged even though the overall run still succeeds (the
            # kernel-log fallback, or the LLM diagnosing the meta-error
            # itself, keeps the run useful) -- ts.tool_ok alone can't
            # express this partial failure since it tracks state.ok.
            "ok": (
                False
                if ts.decision_tool == "discover" and (units_error or containers_error)
                else False
                if ts.decision_tool == "collect" and collect_error
                else ts.tool_ok
            ),
            "duration_ms": ts.duration_ms,
        }
        for ts in state.trace
    ]


def run(target_hint: str | None, question: str | None) -> str:
    """Discover active units/containers, collect logs for target_hint
    (or fall back to kernel logs), and synthesize one diagnosis."""
    state = build().run(
        target_hint or "",
        initial_context={"target_hint": target_hint, "question": question},
    )
    subtrace.publish(_to_sub_steps(state))
    return state.final_output or ""
