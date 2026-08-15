"""
Tests for forge.metrics: per-run inference accounting.

The two things worth pinning here are the reset (a second run must
not inherit the first one's totals) and the None-vs-zero distinction
(a backend that reports nothing must not look like a free run).
"""

import forge.llm as llm_mod
from forge import metrics, trace
from forge.types import Completion, Usage


def teardown_function():
    metrics.clear()


# ---------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------


def test_records_calls_and_tokens():
    metrics.start_run()
    metrics.record(Usage(prompt_tokens=100, completion_tokens=10), 500)
    metrics.record(Usage(prompt_tokens=140, completion_tokens=6), 300)

    snap = metrics.snapshot()
    assert snap["llm_calls"] == 2
    assert snap["llm_ms"] == 800
    assert snap["prompt_tokens"] == 240
    assert snap["completion_tokens"] == 16
    assert snap["total_tokens"] == 256


def test_tracks_the_largest_single_prompt():
    """The total cannot tell one bloated prompt from ten small ones,
    and that is the distinction that matters for compaction."""
    metrics.start_run()
    metrics.record(Usage(prompt_tokens=100), 10)
    metrics.record(Usage(prompt_tokens=4000), 10)
    metrics.record(Usage(prompt_tokens=120), 10)

    assert metrics.snapshot()["max_prompt_tokens"] == 4000


def test_start_run_resets_previous_totals():
    metrics.start_run()
    metrics.record(Usage(prompt_tokens=100, completion_tokens=10), 500)

    metrics.start_run()
    metrics.record(Usage(prompt_tokens=7, completion_tokens=1), 20)

    snap = metrics.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["prompt_tokens"] == 7
    assert snap["llm_ms"] == 20


def test_unreported_tokens_stay_none_while_calls_still_count():
    """A backend that reports no counts must still be visible as having
    been called -- silence about tokens is not silence about cost."""
    metrics.start_run()
    metrics.record(Usage(), 1200)

    snap = metrics.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["llm_ms"] == 1200
    assert snap["total_tokens"] is None
    assert snap["max_prompt_tokens"] is None


def test_partial_reporting_counts_what_it_has():
    metrics.start_run()
    metrics.record(Usage(prompt_tokens=50), 10)
    metrics.record(Usage(completion_tokens=5), 10)

    snap = metrics.snapshot()
    assert snap["prompt_tokens"] == 50
    assert snap["completion_tokens"] == 5


# ---------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------


def test_recording_outside_a_run_is_a_no_op():
    metrics.clear()
    metrics.record(Usage(prompt_tokens=10), 5)  # must not raise
    assert metrics.snapshot() is None


def test_call_llm_records_into_the_open_scope(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")
    monkeypatch.setattr(
        llm_mod.ollama,
        "call",
        lambda url, model, prompt: Completion(
            text="hi", usage=Usage(prompt_tokens=42, completion_tokens=3)
        ),
    )

    metrics.start_run()
    llm_mod.call_llm("hello")

    snap = metrics.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["prompt_tokens"] == 42


# ---------------------------------------------------------------------
# the trace record
# ---------------------------------------------------------------------


class _FakeState:
    """Minimal stand-in for AgentState -- _build_record only reads
    these five fields plus the optional started_at."""

    def __init__(self):
        self.user_input = "hello"
        self.trace = []
        self.final_tool = "chat"
        self.ok = True
        self.error = None


def test_trace_record_carries_the_llm_totals():
    metrics.start_run()
    metrics.record(Usage(prompt_tokens=100, completion_tokens=10), 500)

    record = trace._build_record(_FakeState())
    assert record["llm"]["llm_calls"] == 1
    assert record["llm"]["total_tokens"] == 110


def test_trace_record_omits_llm_when_no_scope_was_opened():
    """A run outside the orchestrator has no measurement -- the field
    must be absent rather than a zeroed one that reads as a free run."""
    metrics.clear()
    record = trace._build_record(_FakeState())
    assert record["llm"] is None


def test_display_shows_the_inference_share_of_the_run():
    traces = [
        {
            "timestamp": "2026-08-15T10:00:00",
            "run_id": "abcd1234",
            "total_ms": 1000,
            "ok": True,
            "steps": [{"router_tool": "chat"}],
            "user_input_preview": "hi",
            "llm": {
                "llm_calls": 2,
                "llm_ms": 900,
                "total_tokens": 256,
            },
        }
    ]
    out = trace.format_for_display(traces)
    assert "2 calls" in out
    assert "90% of run" in out
    assert "256 tokens" in out
