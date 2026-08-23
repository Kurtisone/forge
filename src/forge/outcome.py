"""
Per-run channel for "this turn did not answer the question".

THE PROBLEM OF REACH

A graph-based tool that fails on purpose hides that fact from
everything above it. graphs/recall.py's error node sets `ok = True`
with a comment saying why -- "surface as message, not crash" -- so the
caller gets a clean result instead of an exception. That is the right
behaviour for the conversation and it erases the one thing the store
needs to know: the run ended without an answer, and the exchange about
to be persisted is a question followed by nothing.

The tool contract is `run(content) -> str` and five modules depend on
it (see orchestrator.py's docstring, rule 5). Widening it for this
would touch fourteen tools to serve one. So this follows subtrace.py
and metrics.py: a contextvar side channel, cheaper than a wider
contract, and per-context rather than a module global because api.py
serves chat turns from a two-worker pool and one turn must not read
the other's verdict.

WHY THIS, WHEN forge/non_answer.py ALREADY CATCHES THESE

Today the two overlap almost exactly: the strings recall emits when it
fails are the strings non_answer recognises. They do not overlap in
what makes them break.

non_answer matches on wording. Reword the refusal, or let the model
phrase its own, and it silently stops matching -- the same silent
drift POINTER_RE was nearly lost to, and with the same symptom, which
is nothing at all until a retrieval months later returns something
nobody can explain.

This matches on what the run did. A graph that knows it failed says so
here, and the wording is then free to change. It is also the only one
of the two that could ever catch a refusal written in prose, on the
day a graph is given a state it can report and a sentence it is
allowed to choose.

So: this is the structural half, non_answer is the textual half, and
the textual half is additionally the ONLY half available to
deploy/rag_resplit.py, which works on blocks written down weeks before
any of this existed.

LIFECYCLE, EXPLICIT RATHER THAN LAZY

    orchestrator.run()   -> outcome.clear()        (reset)
    graphs/*.run()       -> outcome.no_answer(...) (report)
    orchestrator._finish -> outcome.taken()        (read, and clear)

clear() at the top of every run for the reason metrics.start_run()
resets rather than creates-if-absent: without it a second run in the
same context inherits the first one's verdict, and every turn after a
failed recall is persisted as unanswered. That is worse than not
marking at all -- it silently stops indexing a working conversation.
"""

from __future__ import annotations

import contextvars

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "forge_outcome_no_answer", default=None
)


def no_answer(reason: str) -> None:
    """
    Called by a graph or tool that is about to return something which
    does not answer the question. `reason` is for the log, not for the
    user -- whatever the run already put in state.error is right.
    """
    _current.set(reason or "unspecified")


def pending() -> str | None:
    """The current verdict without consuming it (logs, tests)."""
    return _current.get()


def taken() -> str | None:
    """
    Read and clear. Called once per run by orchestrator._finish, so
    nothing leaks into the next turn even on the paths that never
    reach clear().
    """
    reason = _current.get()
    _current.set(None)
    return reason


def clear() -> None:
    """Reset. Called at the top of orchestrator.run()."""
    _current.set(None)
