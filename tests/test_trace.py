"""
Tests for forge.trace: the durable JSONL execution record used by
`forge replay` and the `!trace` REPL command.

39% coverage before this file: save()'s TRACE_ENABLED short-circuit
and best-effort failure handling, read_last()'s ordering / corrupt-line
skipping / n-limit, and format_for_display() were all untested.
"""

import json

import forge.trace as trace_mod
from forge.types import AgentState, ToolResult


def _state_with_one_step(ok=True, error=None) -> AgentState:
    state = AgentState(user_input="hello world", max_steps=1)
    ts = state.new_step()
    ts.decision_tool = "chat"
    ts.decision_content = "hi there"
    ts.finish(ToolResult(tool="chat", output="hi there", ok=ok, error=error))
    state.final_output = "hi there"
    state.final_tool = "chat"
    state.ok = ok
    state.error = error
    return state


# ── save() ───────────────────────────────────────────────────────────

def test_save_does_nothing_when_trace_disabled(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setattr(trace_mod, "TRACE_ENABLED", False)
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(trace_file))

    trace_mod.save(_state_with_one_step())

    assert not trace_file.exists()


def test_save_appends_a_json_line(monkeypatch, tmp_path):
    trace_file = tmp_path / "sub" / "traces.jsonl"
    monkeypatch.setattr(trace_mod, "TRACE_ENABLED", True)
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(trace_file))

    trace_mod.save(_state_with_one_step())

    assert trace_file.exists()
    lines = trace_file.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["user_input_preview"] == "hello world"
    assert record["ok"] is True
    assert record["steps"][0]["router_tool"] == "chat"


def test_save_appends_without_clobbering_previous_runs(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setattr(trace_mod, "TRACE_ENABLED", True)
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(trace_file))

    trace_mod.save(_state_with_one_step())
    trace_mod.save(_state_with_one_step())

    assert len(trace_file.read_text().splitlines()) == 2


def test_save_is_best_effort_and_never_raises(monkeypatch, tmp_path):
    """A trace write failure must never break the run it's trying to record."""
    # Point TRACE_FILE at a path whose parent can't be created (a file,
    # not a directory, sitting where a directory is expected).
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setattr(trace_mod, "TRACE_ENABLED", True)
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(blocker / "traces.jsonl"))

    trace_mod.save(_state_with_one_step())  # must not raise


# ── read_last() ───────────────────────────────────────────────────────

def test_read_last_returns_empty_list_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(tmp_path / "does_not_exist.jsonl"))
    assert trace_mod.read_last() == []


def test_read_last_returns_oldest_first_within_the_window(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    trace_file.write_text(
        "\n".join(json.dumps({"run_id": rid}) for rid in ["a", "b", "c"]) + "\n"
    )
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(trace_file))

    result = trace_mod.read_last(2)

    assert [r["run_id"] for r in result] == ["b", "c"]


def test_read_last_skips_corrupt_lines(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    trace_file.write_text(
        json.dumps({"run_id": "good1"}) + "\n"
        "{not valid json\n"
        "\n"  # blank line
        + json.dumps({"run_id": "good2"}) + "\n"
    )
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(trace_file))

    result = trace_mod.read_last(10)

    assert [r["run_id"] for r in result] == ["good1", "good2"]


def test_read_last_handles_unreadable_file_gracefully(monkeypatch, tmp_path):
    trace_file = tmp_path  # a directory, not a file -- read_text() will raise
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(trace_file))
    assert trace_mod.read_last() == []


# ── format_for_display() ────────────────────────────────────────────

def test_format_for_display_empty():
    assert trace_mod.format_for_display([]) == "(no traces yet)"


def test_format_for_display_ok_run():
    traces = [{
        "ok": True,
        "timestamp": "2026-07-01T00:00:00",
        "run_id": "abc123",
        "total_ms": 42,
        "steps": [{"router_tool": "chat"}, {"router_tool": "code"}],
        "user_input_preview": "hello",
    }]
    out = trace_mod.format_for_display(traces)
    assert "✓" in out
    assert "abc123" in out
    assert "chat → code" in out
    assert "'hello'" in out


def test_format_for_display_failed_run_shows_error():
    traces = [{
        "ok": False,
        "timestamp": "2026-07-01T00:00:00",
        "run_id": "err001",
        "total_ms": 5,
        "steps": [],
        "user_input_preview": "boom",
        "error": "provider unavailable",
    }]
    out = trace_mod.format_for_display(traces)
    assert "✗" in out
    assert "no steps" in out
    assert "provider unavailable" in out


def test_format_for_display_ok_run_does_not_show_error_even_if_present():
    """error is only meaningful (and only shown) on failed runs."""
    traces = [{
        "ok": True,
        "timestamp": "t",
        "run_id": "r1",
        "total_ms": 1,
        "steps": [],
        "user_input_preview": "x",
        "error": "stale leftover field",
    }]
    out = trace_mod.format_for_display(traces)
    assert "stale leftover field" not in out
