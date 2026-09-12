"""
Which backend serves the capability currently executing.

THE PROBLEM OF REACH, AGAIN

`FORGE_PROVIDER` is one value for the whole process, so every LLM call
Forge makes -- a routing decision, a research synthesis, a compaction
summary -- goes to the same backend whether or not that is a good idea.
ARCHITECTURE.md's Niveau 2 is written against a world where it is not:
the Router asks for a capability, the Registry lists candidates, the
Scheduler chooses. Nothing has ever had anything to choose between,
because the answer was a module-level constant before the question was
asked.

The natural fix is a parameter on call_llm. That is fourteen call sites
across nine modules, and every graph would have to name itself on every
call -- a wider contract for one feature, and one more thing to forget
when a graph is added. So this follows subtrace.py, metrics.py,
outcome.py and turn.py: a contextvar side channel, set by the one
component that already knows which capability is running, read by the
one that needs it. Per-context rather than a module global because
api.py serves chat turns from a two-worker pool, and one turn must not
be answered by the other's backend.

WHAT THIS IS NOT

It is not the Cognitive Scheduler, and it does not pretend to be. Each
capability still resolves to exactly ONE backend, from configuration,
deterministically -- so kernel/registry.candidates() still returns one
candidate and orchestrator._dispatch's hard stop on an ambiguous
capability keeps meaning what it says. What changes is that "which
backend" stops being a property of the process and becomes a property
of the work, which is the thing a Scheduler would later decide instead
of read.

It also carries no cost, latency or quality scores. kernel/capability.py
states why at length: numbers nobody measured would let a Scheduler look
informed while deciding on fiction. Configuration is a person choosing,
which is honest about being a preference.

FAILING SAFE

An entry naming a backend Forge does not have, or malformed, is dropped
with an error in the log and the global FORGE_PROVIDER answers instead.
The alternative -- refusing to start -- would turn a typo in one
capability into a dead assistant, and this knob is empty by default
precisely because nothing depends on it.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass

from forge.config import CAPABILITY_PROVIDER
from forge.logger import log

#: The backends forge/llm.py knows how to call. A capability pointed at
#: anything else is a typo, and is treated as one.
KNOWN_PROVIDERS = frozenset({"llama_cpp", "ollama", "openrouter"})


@dataclass(frozen=True)
class Target:
    """A backend, and optionally the model name to send it."""

    provider: str
    model: str | None = None


def _parse(spec: str) -> dict[str, Target]:
    """
    `cap=provider` or `cap=provider:model`, comma-separated.

    Parsed once at import. A bad entry is dropped rather than raised on:
    see FAILING SAFE above.
    """
    targets: dict[str, Target] = {}
    for chunk in spec.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        capability, sep, target = entry.partition("=")
        capability, target = capability.strip(), target.strip()
        if not sep or not capability or not target:
            log.error("CAPABILITY_PROVIDER: %r is not `capability=provider`", entry)
            continue
        # partition, not split: an openrouter model name contains
        # slashes and may contain colons, and only the FIRST colon
        # separates the backend from it.
        provider, _, model = target.partition(":")
        provider, model = provider.strip(), model.strip()
        if provider not in KNOWN_PROVIDERS:
            log.error(
                "CAPABILITY_PROVIDER: %r is not a known backend (%s), ignoring %r",
                provider,
                ", ".join(sorted(KNOWN_PROVIDERS)),
                capability,
            )
            continue
        targets[capability] = Target(provider=provider, model=model or None)
    return targets


TARGETS: dict[str, Target] = _parse(CAPABILITY_PROVIDER)

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "forge_serving_capability", default=None
)


@contextmanager
def serving(capability: str):
    """
    Mark the capability being executed, for the duration of the block.

    A context manager rather than a set/clear pair because the thing it
    must survive is an exception: a tool that raises mid-dispatch would
    otherwise leave its name behind, and the next call_llm in the same
    context -- the orchestrator's own next routing decision -- would be
    answered by that tool's backend.
    """
    token = _current.set(capability)
    try:
        yield
    finally:
        _current.reset(token)


def current() -> str | None:
    """The capability being executed, or None outside any dispatch."""
    return _current.get()


def target_for(capability: str | None) -> Target | None:
    """
    The configured backend for this capability, or None for "use the
    global one". None capability means the router's own call, which is
    configurable under the name `router`.
    """
    if not TARGETS:
        return None
    return TARGETS.get(capability or "router")
