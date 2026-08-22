"""
Sub-step side channel for graph-based tools.

Context: the tool contract is strict and deliberate (see
orchestrator.py's module docstring, rule 5) -- `tool.run(content) ->
str`, enforced by _validate_tool_output(). Every existing tool and
test depends on that. But graph-based tools (review, research,
sysadmin) run a Graph internally, and Graph.py *already* records a
rich TraceStep per node (see graph.py's Node.execute) -- that detail
was simply discarded at the tool-wrapper boundary, because the
wrapper only ever returned state.final_output.

Rather than widen the tool contract for every tool, a graph-based
tool can optionally publish its internal steps here right before
returning its string -- a contextvar, not a module-global, so it's
safe under the async request handling in api.py (each request gets
its own context) and self-clears between orchestrator runs.

Usage in a graph-based tool wrapper (e.g. graphs/sysadmin.run):
    from forge import subtrace
    state = build().run(...)
    subtrace.publish([
        {"label": "discover", "detail": "2 units, 1 container", "duration_ms": 12},
        ...
    ])
    return state.final_output

The orchestrator reads it back with pop() immediately after calling
the tool, attaches it to the ToolResult, and clears it -- so a tool
that never publishes anything (the vast majority) costs nothing and
sees no behavior change at all.
"""

from __future__ import annotations

import contextvars

_current: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "forge_subtrace_current", default=None
)


def publish(steps: list[dict]) -> None:
    """Called by a graph-based tool right before returning its output
    string. `steps` should be small, JSON-serializable dicts -- see
    graphs/sysadmin.py's _to_sub_steps for the expected shape
    ({"label", "detail", "ok", "duration_ms"})."""
    _current.set(steps)


def pop() -> list[dict] | None:
    """Read and clear whatever the just-completed tool call published.
    Called once by orchestrator._dispatch() right after the tool
    returns, so nothing leaks into the next tool's dispatch."""
    steps = _current.get()
    _current.set(None)
    return steps


def clear() -> None:
    """Reset the channel to empty. Called by orchestrator._dispatch()
    right BEFORE invoking a tool -- without this, a tool that never
    publishes anything could silently inherit whatever the previous
    dispatch (or a leftover from outside this call, e.g. a test) left
    behind, since pop() alone only clears AFTER reading."""
    _current.set(None)


def from_state(state, details: dict | None = None) -> list[dict]:
    """
    Turn a finished graph run's internal trace into publishable steps.

    Every graph already records a TraceStep per node -- graph.py's
    Node.execute has done that since the engine shipped. Only sysadmin
    ever published them, so `review`, `research`, `recall` and
    `delegate` ran three to five nodes each and showed the user a
    single opaque box, while the identical information sat on
    `state.trace` and was dropped at the wrapper boundary.

    `details` maps a node name to a zero-argument callable returning a
    one-line description. It is optional and per-graph on purpose: the
    node NAMES are already meaningful ("read_file", "search",
    "synthesize"), so a graph that supplies nothing still gets a real
    timeline, and a graph with something worth saying can say it
    without every other graph paying for the machinery.

    A callable rather than a string because the detail is computed
    from the finished state, and a dict of strings would have to be
    built eagerly for nodes the run never reached.
    """
    details = details or {}
    return [
        {
            "label": step.decision_tool,
            "detail": details.get(step.decision_tool, lambda: "")(),
            "ok": step.tool_ok,
            "duration_ms": step.duration_ms,
        }
        for step in state.trace
    ]
