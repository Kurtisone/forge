"""
Per-run channel for "this turn was about delegation job N".

THE PROBLEM OF REACH, ONCE MORE

A delegation turn already knows the job id. What it hands back is a
sentence -- `f"Job {job.id} lancé."` -- and the id exists nowhere else
in the result, so a client that needs to follow the job in the
background has to read it back out of French prose with a regex. That
works until one of those five strings is reworded, which is a change
nobody would think to look for from the other side of an HTTP
boundary.

The id has to travel from two places that share no plumbing:

  - delegation.intercept(), whose contract is `str | None` and whose
    callers never see a job at all;
  - graphs/delegate.py, which reaches the orchestrator through
    AgentState like every other graph.

Widening the first means changing a contract to carry a value that is
None on almost every call. So this follows outcome.py, subtrace.py,
metrics.py, turn.py and serving.py: a contextvar side channel, set by
whoever already knows, read once by the single exit path. Per-context
rather than a module global because api.py serves chat turns from a
two-worker pool, and one turn must not report the other's job.

WHAT COUNTS AS "ABOUT" A JOB

Any turn that names one: creating it, answering one of its questions,
approving it, launching it, cancelling it. Not just the ones that
change its state -- a turn that re-asks a question it already asked is
still a turn about that job, and a client watching job 12 gains
nothing from being told this exchange was about no job at all.

The one delegation turn that reports nothing is `jobs`, the listing:
it is about all of them, which is the same as about none in a field
that holds one id.

LIFECYCLE, EXPLICIT RATHER THAN LAZY

    orchestrator.run()   -> current_job.clear()   (reset)
    delegation/delegate  -> current_job.report(id) (report)
    orchestrator._finish -> current_job.taken()   (read, and clear)

clear() at the top of every run for the reason outcome.clear() exists:
without it, the turn after a delegation inherits its id and every
later answer arrives tagged with a job it has nothing to do with.
"""

from __future__ import annotations

import contextvars

_current: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "forge_current_job_id", default=None
)


def report(job_id: int) -> None:
    """Called by whichever delegation path already holds the job."""
    _current.set(job_id)


def pending() -> int | None:
    """The current id without consuming it (logs, tests)."""
    return _current.get()


def taken() -> int | None:
    """
    Read and clear. Called once per run by orchestrator._finish, so
    nothing leaks into the next turn even on the paths that never
    reach clear().
    """
    job_id = _current.get()
    _current.set(None)
    return job_id


def clear() -> None:
    """Reset. Called at the top of orchestrator.run()."""
    _current.set(None)
