"""
Forge sysadmin graph: discover -> collect -> synthesize.

Same reasoning as graphs/research.py: a deterministic fixed sequence,
never a router-driven multi-step chain. The router makes exactly ONE
decision (call "sysadmin"), the graph itself decides what to run next
-- no mid-flow judgment call is ever handed back to the model.

Security model (read-only, always):
  - discover_node runs two fixed, parameter-free commands
    (systemctl list-units, podman ps) -- nothing here can be
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

import subprocess

from forge.config import (
    SYSADMIN_COLLECT_TIMEOUT,
    SYSADMIN_DISCOVERY_TIMEOUT,
    SYSADMIN_LOG_CHARS_BUDGET,
    SYSADMIN_MAX_LOG_LINES,
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
_PROMPT_LEAK_MARKERS = [
    "Respond in plain text",
    "GOOD ANSWER:",
    "NEVER DO THIS",
]

# Fixed, parameter-free discovery commands.
_DISCOVER_UNITS_CMD = [
    "systemctl",
    "list-units",
    "--type=service",
    "--state=running",
    "--no-pager",
    "--no-legend",
]
_DISCOVER_CONTAINERS_CMD = ["podman", "ps", "--format", "{{.Names}}"]

# {name} is substituted only after collect_node has verified the name
# against discover_node's own output -- see collect_node's docstring.
_COLLECT_TEMPLATES = {
    "unit": ["journalctl", "-u", "{name}", "--no-pager", "-n", str(SYSADMIN_MAX_LOG_LINES)],
    "container": ["podman", "logs", "--tail", str(SYSADMIN_MAX_LOG_LINES), "{name}"],
    "kernel": ["journalctl", "-k", "--no-pager", "-n", str(SYSADMIN_MAX_LOG_LINES)],
}

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

GOOD ANSWER: Le service searxng redémarre en boucle car le port 8888
est déjà occupé au démarrage d'après les lignes "address already in
use". Je te propose de vérifier quel processus occupe ce port avant
de relancer le service.
NEVER DO THIS: {{"tool":"chat","content":"..."}}

Now write your own diagnosis for the question above, in the same
plain format as GOOD ANSWER -- not the NEVER DO THIS shape. Be
concise.
"""


def _run_fixed(cmd: list[str], timeout: int) -> str:
    """Run a command whose every element is either a fixed literal or
    a name already verified against discover_node's own output.
    Never shell=True, never a hand-built string -- same posture as
    tools/shell.py's allowlisted subprocess.run(parts, ...)."""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return f"[error] executable not found: {cmd[0]!r}"
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except OSError as e:
        return f"[error] OS error: {e}"

    output = (result.stdout or result.stderr or "").strip()
    lines = output.splitlines()[:SYSADMIN_MAX_LOG_LINES]
    return "\n".join(lines) if lines else "[no output]"


def _discover_node(state: AgentState) -> AgentState:
    units_raw = _run_fixed(_DISCOVER_UNITS_CMD, SYSADMIN_DISCOVERY_TIMEOUT)
    units = [line.split()[0] for line in units_raw.splitlines() if line.split()]

    containers_raw = _run_fixed(_DISCOVER_CONTAINERS_CMD, SYSADMIN_DISCOVERY_TIMEOUT)
    containers = [line.strip() for line in containers_raw.splitlines() if line.strip()]

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
        cmd = [p.format(name=target_hint) for p in _COLLECT_TEMPLATES["unit"]]
        source = f"journalctl -u {target_hint}"
    elif target_hint and target_hint in containers:
        cmd = [p.format(name=target_hint) for p in _COLLECT_TEMPLATES["container"]]
        source = f"podman logs {target_hint}"
    else:
        if target_hint:
            log.warning(
                "sysadmin: target_hint %r not found in discovery, falling back to kernel logs",
                target_hint,
            )
        cmd = _COLLECT_TEMPLATES["kernel"]
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
    question = state.context.get("question") or "Diagnostique le problème et propose une solution."
    source = state.context["log_source"]
    log_block = _truncate_log_block(state.context["collected_logs"], SYSADMIN_LOG_CHARS_BUDGET)

    prompt = _SYNTHESIS_PROMPT.format(
        today_line=today_line(),
        question=question,
        source=source,
        log_block=log_block,
    )

    log.event("sysadmin.llm_call", source=source, prompt_chars=len(prompt))
    try:
        raw = call_llm(prompt)
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    log.event("sysadmin.raw_output", raw=raw)
    state.final_output = _clean_diagnosis_response(raw)
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


def run(target_hint: str | None, question: str | None) -> str:
    """Discover active units/containers, collect logs for target_hint
    (or fall back to kernel logs), and synthesize one diagnosis."""
    state = build().run(
        target_hint or "",
        initial_context={"target_hint": target_hint, "question": question},
    )
    return state.final_output or ""
