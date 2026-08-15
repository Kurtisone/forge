"""
Per-run inference accounting.

The problem this solves is one of reach, not of measurement. Usage is
known at the provider boundary (see types.Completion), but the object
that would naturally hold it -- AgentState -- is four frames up and
sometimes on the other side of a Graph: compaction, the router, and
every graph node call forge.llm.call_llm directly, with no state in
hand and no way to thread one through without widening a contract
that five modules depend on.

So the totals accumulate in a contextvar, the same shape subtrace.py
uses for graph sub-steps and for the same reason: a side channel is
cheaper than a wider contract, and a contextvar (not a module global)
keeps concurrent API requests from summing into each other's totals.

The lifecycle is deliberately explicit rather than lazy:

    orchestrator.run()      -> metrics.start_run()   (reset)
    llm.call_llm()          -> metrics.record(...)   (accumulate)
    trace._build_record()   -> metrics.snapshot()    (read)

start_run() resets rather than merely creating-if-absent. Without it,
a second run in the same REPL context inherits the first one's totals
and every run after the first reports a number that only grows -- the
exact trap subtrace.clear() exists to avoid.

Accounting is best-effort and never load-bearing: a run whose backend
reports no counts records calls and milliseconds and leaves the token
totals at None. It must never be the reason a run fails.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field

from forge.types import Usage


@dataclass
class RunMetrics:
    """Inference totals for one run(). Mutable: filled in as it goes."""

    llm_calls: int = 0
    llm_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Per-call prompt sizes, in order. The total alone cannot
    # distinguish one huge prompt from ten small ones, and that
    # distinction is the whole question when deciding whether a run is
    # expensive because the context is bloated or because it took many
    # steps.
    prompt_sizes: list[int] = field(default_factory=list)

    def add(self, usage: Usage, elapsed_ms: int) -> None:
        self.llm_calls += 1
        self.llm_ms += elapsed_ms
        if usage.prompt_tokens is not None:
            self.prompt_tokens = (self.prompt_tokens or 0) + usage.prompt_tokens
            self.prompt_sizes.append(usage.prompt_tokens)
        if usage.completion_tokens is not None:
            self.completion_tokens = (
                self.completion_tokens or 0
            ) + usage.completion_tokens

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "llm_ms": self.llm_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "max_prompt_tokens": max(self.prompt_sizes) if self.prompt_sizes else None,
        }


_current: contextvars.ContextVar[RunMetrics | None] = contextvars.ContextVar(
    "forge_metrics_current", default=None
)


def start_run() -> RunMetrics:
    """Open a fresh accounting scope. Called by orchestrator.run()."""
    m = RunMetrics()
    _current.set(m)
    return m


def record(usage: Usage, elapsed_ms: int) -> None:
    """
    Add one completion to the current run's totals.

    A no-op outside a run scope, on purpose: call_llm is reachable from
    tests, from the CLI, and from tooling that never opened one, and
    none of those should have to know this module exists.
    """
    m = _current.get()
    if m is None:
        return
    m.add(usage, elapsed_ms)


def snapshot() -> dict | None:
    """Current totals as a plain dict, or None outside a run scope."""
    m = _current.get()
    return m.to_dict() if m is not None else None


def clear() -> None:
    """Close the accounting scope. Mainly for tests and for symmetry."""
    _current.set(None)
