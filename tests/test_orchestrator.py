"""
Smoke tests for the orchestrator. No network: the LLM call is
monkeypatched at the boundary (forge.orchestrator.call_llm).
"""

import json

import forge.orchestrator as orch_mod
from forge.orchestrator import Orchestrator


def test_chat_round_trip(monkeypatch):
    monkeypatch.setattr(
        orch_mod, "call_llm", lambda prompt: json.dumps({"tool": "chat", "content": "hi there"})
    )
    result = Orchestrator().run("hello")
    assert result.ok
    assert result.tool == "chat"
    assert result.output == "hi there"
    assert result.steps == 1


def test_code_round_trip(monkeypatch):
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "code", "content": "print(1)"}),
    )
    result = Orchestrator().run("write code")
    assert result.ok
    assert result.tool == "code"
    assert "print(1)" in result.output


def test_malformed_router_output_falls_back_to_chat(monkeypatch):
    monkeypatch.setattr(orch_mod, "call_llm", lambda prompt: "not json at all")
    result = Orchestrator().run("hello")
    assert result.ok
    assert result.tool == "chat"
    assert result.output == "not json at all"


def test_runaway_local_model_output_still_parses(monkeypatch):
    """
    Regression test for a real-world failure: a local llama.cpp model
    kept generating fake "User: ..." turns after the JSON object
    because the old stop sequences never matched. The parser must
    extract the leading JSON object instead of discarding the whole
    answer just because of trailing garbage.
    """
    raw = (
        '{"tool":"chat","content":"answer"}\n'
        "User: some question\n"
        '{"tool":"chat","content":"answer"}\n'
        "User: some question\n"
    )
    monkeypatch.setattr(orch_mod, "call_llm", lambda prompt: raw)
    result = Orchestrator().run("some question")
    assert result.ok
    assert result.tool == "chat"
    assert result.output == "answer"


def test_unknown_tool_falls_back_to_chat(monkeypatch):
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "shell", "content": "rm -rf /"}),
    )
    result = Orchestrator().run("do something dangerous")
    assert result.tool == "chat"  # shell isn't registered, parser/orchestrator fall back


def test_provider_failure_is_reported_not_raised(monkeypatch):
    from forge.errors import ProviderError

    def boom(prompt):
        raise ProviderError("backend down")

    monkeypatch.setattr(orch_mod, "call_llm", boom)
    result = Orchestrator().run("hello")
    assert not result.ok
    assert "backend down" in result.error


def test_max_steps_is_respected_even_at_zero(monkeypatch):
    monkeypatch.setattr(orch_mod, "call_llm", lambda prompt: '{"tool":"chat","content":"x"}')
    import pytest

    with pytest.raises(Exception):
        Orchestrator(max_steps=0).run("hello")
