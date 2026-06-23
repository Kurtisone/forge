"""
Tests for the strict ToolResult contract: a tool handler MUST return
a non-empty str. Anything else is a contract violation that produces
a typed, failed AgentResult -- never a crash, never a confusing
non-string value passed through to the user.
"""

import json

import forge.orchestrator as orch_mod
from forge.orchestrator import Orchestrator
from forge.tools.registry import TOOLS


def _run_with_fake_chat_handler(monkeypatch, fake_handler):
    monkeypatch.setitem(TOOLS, "chat", fake_handler)
    monkeypatch.setattr(
        orch_mod, "call_llm", lambda p: json.dumps({"tool": "chat", "content": "x"})
    )
    return Orchestrator().run("hello")


def test_none_output_is_a_contract_violation(monkeypatch):
    result = _run_with_fake_chat_handler(monkeypatch, lambda content: None)
    assert not result.ok
    assert "must return str" in result.error


def test_non_str_output_is_a_contract_violation(monkeypatch):
    result = _run_with_fake_chat_handler(monkeypatch, lambda content: {"oops": "dict"})
    assert not result.ok
    assert "must return str" in result.error


def test_int_output_is_a_contract_violation(monkeypatch):
    result = _run_with_fake_chat_handler(monkeypatch, lambda content: 42)
    assert not result.ok
    assert "must return str" in result.error


def test_whitespace_only_output_is_a_contract_violation(monkeypatch):
    result = _run_with_fake_chat_handler(monkeypatch, lambda content: "   ")
    assert not result.ok
    assert "empty output" in result.error


def test_well_behaved_tool_still_passes(monkeypatch):
    result = _run_with_fake_chat_handler(monkeypatch, lambda content: "real answer")
    assert result.ok
    assert result.output == "real answer"
