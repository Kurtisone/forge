"""
Tests for forge.graph and forge.graphs.default.

All tests use pure Python node functions — no LLM calls, no network.
The default graph tests monkeypatch forge.graphs.default.call_llm,
the module-level name where the function is actually bound.
"""

import json

import forge.graphs.default as default_mod
from forge.errors import ProviderError
from forge.graph import Graph
from forge.graphs.default import build
from forge.tools.registry import load_tools


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _identity(state):
    return state


def _set_output(value):
    def fn(state):
        state.final_output = value
        return state
    return fn


def _fail(state):
    state.ok = False
    state.error = "intentional"
    state.final_output = "failed"
    return state


def _recover(state):
    state.final_output = "recovered"
    state.ok = True
    state.error = None
    return state


# -------------------------------------------------------------------
# Graph engine tests
# -------------------------------------------------------------------

def test_single_node_executes():
    g = Graph("t")
    g.add_node("a", _set_output("done"))
    state = g.run("hello")
    assert state.ok
    assert state.final_output == "done"
    assert state.steps_taken == 1


def test_two_node_chain():
    def upper(s):
        s.final_output = s.user_input.upper()
        return s

    def exclaim(s):
        s.final_output = s.final_output + "!"
        return s

    g = Graph("t")
    g.add_node("upper", upper)
    g.add_node("exclaim", exclaim)
    g.add_edge("upper", "exclaim")
    state = g.run("hello")
    assert state.final_output == "HELLO!"
    assert state.steps_taken == 2


def test_conditional_edge_on_failure():
    g = Graph("t")
    g.add_node("work", _fail)
    g.add_node("recover", _recover)
    g.add_edge("work", "recover", condition=lambda s: not s.ok)
    state = g.run("x")
    assert state.final_output == "recovered"
    assert state.ok
    assert state.steps_taken == 2


def test_conditional_edge_skipped_on_success():
    g = Graph("t")
    g.add_node("work", _set_output("ok"))
    g.add_node("recover", _recover)
    g.add_edge("work", "recover", condition=lambda s: not s.ok)
    state = g.run("x")
    assert state.final_output == "ok"
    assert state.steps_taken == 1


def test_max_steps_guard():
    g = Graph("t", max_steps=3)
    g.add_node("a", _identity)
    g.add_node("b", _identity)
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    state = g.run("loop")
    assert not state.ok
    assert "max_steps" in (state.error or "")


def test_trace_recorded_per_node():
    def upper(s):
        s.final_output = s.user_input.upper()
        return s

    def exclaim(s):
        s.final_output = s.final_output + "!"
        return s

    g = Graph("t")
    g.add_node("upper", upper)
    g.add_node("exclaim", exclaim)
    g.add_edge("upper", "exclaim")
    state = g.run("hi")
    assert len(state.trace) == 2
    assert state.trace[0].decision_tool == "upper"
    assert state.trace[1].decision_tool == "exclaim"
    assert state.trace[0].duration_ms is not None


def test_unknown_node_in_edge_raises():
    import pytest
    g = Graph("t")
    g.add_node("a", _identity)
    with pytest.raises(ValueError, match="not registered"):
        g.add_edge("a", "missing")


# -------------------------------------------------------------------
# Default graph tests
# -------------------------------------------------------------------

def test_default_graph_successful_run(monkeypatch):
    load_tools()
    monkeypatch.setattr(
        default_mod, "call_llm",
        lambda p: json.dumps({"tool": "chat", "content": "Bonjour !"})
    )
    state = build().run("Salut")
    assert state.ok
    assert state.final_output == "Bonjour !"
    assert state.steps_taken == 2
    assert state.trace[0].decision_tool == "router"
    assert state.trace[1].decision_tool == "dispatch"


def test_default_graph_provider_failure_triggers_fallback(monkeypatch):
    load_tools()
    monkeypatch.setattr(
        default_mod, "call_llm",
        lambda p: (_ for _ in ()).throw(ProviderError("down"))
    )
    state = build().run("hello")
    assert state.ok        # fallback recovered
    assert "went wrong" in (state.final_output or "")
    assert state.steps_taken == 3   # router + dispatch + fallback


def test_default_graph_to_result(monkeypatch):
    load_tools()
    monkeypatch.setattr(
        default_mod, "call_llm",
        lambda p: json.dumps({"tool": "chat", "content": "hi"})
    )
    result = build().run("hello").to_result()
    assert result.ok
    assert result.output == "hi"
    assert result.steps == 2
