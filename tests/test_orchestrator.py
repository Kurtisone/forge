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


def test_leaked_role_prefix_is_stripped(monkeypatch):
    """
    Regression test: once conversation history was added to the
    router prompt, some local models started leaking 'Assistant: ...'
    style prefixes into otherwise-non-JSON output. The label must not
    end up visible in the final answer.
    """
    monkeypatch.setattr(
        orch_mod, "call_llm", lambda prompt: "Assistant: here is my answer"
    )
    result = Orchestrator().run("hello")
    assert result.ok
    assert result.output == "here is my answer"


def test_history_is_passed_as_context_not_dialogue(monkeypatch):
    """
    Regression test: the history block used to be formatted as
    'User: ... / Assistant: ...', which visually matched the live
    turn prompt and caused local models to continue it as plain
    dialogue instead of emitting JSON. It must read as reference
    context instead. (Memory file isolation comes from the autouse
    fixture in conftest.py.)
    """
    captured = {}

    def capture_and_answer(prompt):
        captured["prompt"] = prompt
        return json.dumps({"tool": "chat", "content": "ok"})

    monkeypatch.setattr(orch_mod, "call_llm", lambda prompt: json.dumps(
        {"tool": "chat", "content": "Salut Alexandre !"}
    ))
    Orchestrator().run("Je m'appelle Alexandre")

    monkeypatch.setattr(orch_mod, "call_llm", capture_and_answer)
    Orchestrator().run("Comment je m'appelle ?")

    assert "they said" in captured["prompt"]
    assert "\nUser: Je m'appelle Alexandre\n" not in captured["prompt"]


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
