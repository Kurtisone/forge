"""
Tests for forge.llm: the single dispatch point between the
orchestrator and every provider (ollama / llama.cpp / openrouter).

This module had 26% coverage before this file: the provider routing
itself, the unknown-provider error path, and the "wrap any surprise
into ProviderError" path were all untested. A typo in FORGE_PROVIDER
matching, or an exception silently swallowed instead of wrapped,
would have broken every run without a single test noticing.
"""

import pytest

import forge.llm as llm_mod
from forge.errors import ProviderError
from forge.types import Completion, Usage


def test_dispatches_to_ollama(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")
    called = {}

    def fake_call(url, model, prompt):
        called["args"] = (url, model, prompt)
        return Completion(text="ollama says hi")

    monkeypatch.setattr(llm_mod.ollama, "call", fake_call)
    result = llm_mod.call_llm("hello")

    assert result == "ollama says hi"
    assert called["args"][2] == "hello"


def test_dispatches_to_llama_cpp(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "llama_cpp")
    monkeypatch.setattr(
        llm_mod.llama_cpp,
        "call",
        lambda url, model, prompt, grammar=None: Completion(
            text="llama.cpp says hi"
        ),
    )

    result = llm_mod.call_llm("hello")
    assert result == "llama.cpp says hi"


def test_dispatches_to_openrouter(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "openrouter")
    monkeypatch.setattr(
        llm_mod.openrouter,
        "call",
        lambda url, key, model, prompt: Completion(text="openrouter says hi"),
    )

    result = llm_mod.call_llm("hello")
    assert result == "openrouter says hi"


def test_unknown_provider_raises_provider_error(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "not_a_real_provider")

    with pytest.raises(ProviderError, match="Unknown provider"):
        llm_mod.call_llm("hello")


def test_provider_error_propagates_unwrapped(monkeypatch):
    """A ProviderError raised by a provider must reach the caller as-is,
    not get double-wrapped into a generic 'unexpected provider failure'."""
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")

    def raise_provider_error(url, model, prompt):
        raise ProviderError("ollama is down")

    monkeypatch.setattr(llm_mod.ollama, "call", raise_provider_error)

    with pytest.raises(ProviderError, match="ollama is down"):
        llm_mod.call_llm("hello")


def test_unexpected_exception_gets_wrapped_into_provider_error(monkeypatch):
    """Anything a provider throws that ISN'T already a ProviderError must
    still surface as one -- callers only ever need to catch ProviderError."""
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")

    def raise_weird_error(url, model, prompt):
        raise ValueError("connection reset by peer")

    monkeypatch.setattr(llm_mod.ollama, "call", raise_weird_error)

    with pytest.raises(ProviderError, match="unexpected provider failure"):
        llm_mod.call_llm("hello")


# ---------------------------------------------------------------------
# usage passthrough
# ---------------------------------------------------------------------


def test_call_llm_returns_text_not_the_completion(monkeypatch):
    """The orchestrator contract is unchanged: call_llm hands back a
    plain str. Only the provider boundary got richer."""
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")
    monkeypatch.setattr(
        llm_mod.ollama,
        "call",
        lambda url, model, prompt: Completion(
            text="answer", usage=Usage(prompt_tokens=120, completion_tokens=8)
        ),
    )
    result = llm_mod.call_llm("hello")
    assert result == "answer"
    assert isinstance(result, str)


def test_usage_reaches_the_response_log(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")
    monkeypatch.setattr(
        llm_mod.ollama,
        "call",
        lambda url, model, prompt: Completion(
            text="answer", usage=Usage(prompt_tokens=120, completion_tokens=8)
        ),
    )

    events = []
    monkeypatch.setattr(
        llm_mod.log, "event", lambda name, **kw: events.append((name, kw))
    )
    llm_mod.call_llm("hello")

    response = next(kw for name, kw in events if name == "llm.response")
    assert response["prompt_tokens"] == 120
    assert response["completion_tokens"] == 8


def test_grammar_is_forwarded_to_llama_cpp(monkeypatch):
    """
    A per-call grammar is the whole point of v3.13 lot 1: before it,
    llama_cpp.call built the router's grammar unconditionally, so a
    graph asking for a spec-shaped answer got a routing decision's
    shape instead (which is why every graph carries
    try_unwrap_router_json).
    """
    seen = {}

    def _fake_call(url, model, prompt, grammar=None):
        seen["grammar"] = grammar
        return Completion(text="ok")

    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "llama_cpp")
    monkeypatch.setattr(llm_mod.llama_cpp, "call", _fake_call)

    llm_mod.call_llm("hello", grammar='root ::= "x"')
    assert seen["grammar"] == 'root ::= "x"'


def test_grammar_request_is_logged_when_provider_cannot_honour_it(monkeypatch, caplog):
    """
    Ollama and OpenRouter cannot constrain sampling to a schema. The
    call still runs -- refusing would make a provider swap fail on a
    feature the caller can survive without -- but the caller's parse
    is then unprotected, so the downgrade must not be silent.
    """
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")
    monkeypatch.setattr(
        llm_mod.ollama, "call", lambda url, model, prompt: Completion(text="hi")
    )

    with caplog.at_level("WARNING"):
        assert llm_mod.call_llm("hello", grammar='root ::= "x"') == "hi"

    assert "unconstrained" in caplog.text
