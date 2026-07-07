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


def test_dispatches_to_ollama(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "ollama")
    called = {}

    def fake_call(url, model, prompt):
        called["args"] = (url, model, prompt)
        return "ollama says hi"

    monkeypatch.setattr(llm_mod.ollama, "call", fake_call)
    result = llm_mod.call_llm("hello")

    assert result == "ollama says hi"
    assert called["args"][2] == "hello"


def test_dispatches_to_llama_cpp(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "llama_cpp")
    monkeypatch.setattr(llm_mod.llama_cpp, "call", lambda url, model, prompt: "llama.cpp says hi")

    result = llm_mod.call_llm("hello")
    assert result == "llama.cpp says hi"


def test_dispatches_to_openrouter(monkeypatch):
    monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "openrouter")
    monkeypatch.setattr(
        llm_mod.openrouter,
        "call",
        lambda url, key, model, prompt: "openrouter says hi",
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
