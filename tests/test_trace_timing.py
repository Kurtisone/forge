"""
Tests for what a trace reports about time.

Regression cover for a real run (#ef434323) that took two routing
calls of 68s and 73s and reported total_ms=68967 -- roughly half. The
second step exited on the loop guard before dispatch, so it never
reached TraceStep.finish(), so its duration stayed None, so the old
sum-of-steps total silently dropped it. The UI then rendered that
None as the literal string "nullms".
"""

import json

import forge.trace as trace_mod
from forge.types import AgentState, ToolResult


def _enable(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setattr(trace_mod, "TRACE_ENABLED", True)
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(trace_file))
    return trace_file


def _record(trace_file) -> dict:
    return json.loads(trace_file.read_text(encoding="utf-8").strip())


# ── TraceStep.abandon() ──────────────────────────────────────────────


def test_abandon_records_a_duration_and_the_reason():
    state = AgentState(user_input="q", max_steps=2)
    ts = state.new_step()
    ts.abandon("loop guard: repeated call")

    assert ts.duration_ms is not None
    assert ts.duration_ms >= 0
    assert ts.tool_error == "loop guard: repeated call"
    # Not a tool failure: the tool was never dispatched at all.
    assert ts.tool_ok is None


# ── total_ms ─────────────────────────────────────────────────────────


def test_total_ms_covers_a_step_that_never_dispatched(monkeypatch, tmp_path):
    trace_file = _enable(monkeypatch, tmp_path)

    state = AgentState(user_input="list my hardware", max_steps=2)
    state.started_at -= 5.0  # pretend the run began 5s ago

    first = state.new_step()
    first.decision_tool = "memory"
    first.finish(ToolResult(tool="memory", output="- [fact] a Steam Deck"))

    second = state.new_step()
    second.decision_tool = "memory"
    second.abandon("loop guard: repeated call, fell back to previous result")

    state.final_output = "- [fact] a Steam Deck"
    state.final_tool = "memory"
    trace_mod.save(state)

    record = _record(trace_file)
    # The old sum-of-steps total would have been ~0 here, since neither
    # step did any real work; wall clock sees the whole 5s.
    assert record["total_ms"] >= 5000
    assert [s["duration_ms"] is not None for s in record["steps"]] == [True, True]


def test_total_ms_is_not_the_sum_of_step_durations(monkeypatch, tmp_path):
    trace_file = _enable(monkeypatch, tmp_path)

    state = AgentState(user_input="q", max_steps=1)
    state.started_at -= 3.0
    ts = state.new_step()
    ts.finish(ToolResult(tool="chat", output="a"))
    ts.duration_ms = 1  # a step that reported almost nothing

    trace_mod.save(state)

    assert _record(trace_file)["total_ms"] >= 3000


def test_total_ms_falls_back_to_summing_without_a_start_time(monkeypatch, tmp_path):
    """Old traces / hand-built states without started_at still total up."""
    trace_file = _enable(monkeypatch, tmp_path)

    class _Legacy:
        started_at = None
        user_input = "q"
        final_tool = "chat"
        ok = True
        error = None

        def __init__(self):
            state = AgentState(user_input="q", max_steps=1)
            ts = state.new_step()
            ts.decision_tool = "chat"
            ts.finish(ToolResult(tool="chat", output="a"))
            ts.duration_ms = 42
            self.trace = state.trace

    trace_mod.save(_Legacy())

    assert _record(trace_file)["total_ms"] == 42
