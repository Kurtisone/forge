"""
Tests for forge.providers.*. requests.post is monkeypatched, so
these run with no network access and no real LLM backend.
"""

import requests

from forge.errors import ProviderError
from forge.providers import llama_cpp, ollama, openrouter


class FakeResponse:
    def __init__(self, json_data, status_ok=True):
        self._json = json_data
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise requests.HTTPError("simulated HTTP error")

    def json(self):
        return self._json


# ---------------------------------------------------------------------
# llama_cpp
# ---------------------------------------------------------------------

def test_llama_cpp_call_returns_content(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **kw: FakeResponse({"content": "hello"})
    )
    assert llama_cpp.call("http://fake", "model", "prompt") == "hello"


def test_llama_cpp_call_accepts_completion_key(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **kw: FakeResponse({"completion": "hi"})
    )
    assert llama_cpp.call("http://fake", "model", "prompt") == "hi"


def test_llama_cpp_empty_content_raises_provider_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse({}))
    try:
        llama_cpp.call("http://fake", "model", "prompt")
        assert False, "expected ProviderError"
    except ProviderError:
        pass


def test_llama_cpp_network_failure_raises_provider_error(monkeypatch):
    def boom(*a, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    try:
        llama_cpp.call("http://fake", "model", "prompt")
        assert False, "expected ProviderError"
    except ProviderError:
        pass


def test_llama_cpp_http_error_raises_provider_error(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **kw: FakeResponse({}, status_ok=False)
    )
    try:
        llama_cpp.call("http://fake", "model", "prompt")
        assert False, "expected ProviderError"
    except ProviderError:
        pass


def test_llama_cpp_sends_grammar_by_default(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"content": "hi"})

    monkeypatch.setattr(requests, "post", fake_post)
    llama_cpp.call("http://fake", "model", "prompt")

    assert "grammar" in captured["payload"]
    assert "root" in captured["payload"]["grammar"]


def test_llama_cpp_grammar_reflects_enabled_tools(monkeypatch):
    import forge.tools.registry as registry_mod

    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "shell"])
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"content": "hi"})

    monkeypatch.setattr(requests, "post", fake_post)
    llama_cpp.call("http://fake", "model", "prompt")

    assert '"\\"shell\\""' in captured["payload"]["grammar"]
    assert '"\\"code\\""' not in captured["payload"]["grammar"]


def test_llama_cpp_grammar_disabled_via_config(monkeypatch):
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_USE_GRAMMAR", False)
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"content": "hi"})

    monkeypatch.setattr(requests, "post", fake_post)
    llama_cpp.call("http://fake", "model", "prompt")

    assert "grammar" not in captured["payload"]


def test_llama_cpp_still_sends_stop_sequences_with_grammar_enabled(monkeypatch):
    """Grammar is the primary safeguard now, but the stop sequences
    stay as defense-in-depth -- e.g. against a model that doesn't stop
    generating right after a complete, valid object."""
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"content": "hi"})

    monkeypatch.setattr(requests, "post", fake_post)
    llama_cpp.call("http://fake", "model", "prompt")

    assert "User:" in captured["payload"]["stop"]


# ---------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------

def test_ollama_call_returns_response_field(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **kw: FakeResponse({"response": "hello"})
    )
    assert ollama.call("http://fake", "model", "prompt") == "hello"


def test_ollama_empty_response_raises_provider_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse({}))
    try:
        ollama.call("http://fake", "model", "prompt")
        assert False, "expected ProviderError"
    except ProviderError:
        pass


# ---------------------------------------------------------------------
# openrouter
# ---------------------------------------------------------------------

def test_openrouter_missing_api_key_raises_provider_error():
    try:
        openrouter.call("http://fake", "", "model", "prompt")
        assert False, "expected ProviderError"
    except ProviderError as e:
        assert "API_KEY" in str(e)


def test_openrouter_call_returns_message_content(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse(
            {"choices": [{"message": {"content": "hello"}}]}
        ),
    )
    assert openrouter.call("http://fake", "sk-fake", "model", "prompt") == "hello"


def test_openrouter_error_field_raises_provider_error(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse({"error": {"message": "bad request"}}),
    )
    try:
        openrouter.call("http://fake", "sk-fake", "model", "prompt")
        assert False, "expected ProviderError"
    except ProviderError:
        pass


def test_openrouter_missing_choices_raises_provider_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse({}))
    try:
        openrouter.call("http://fake", "sk-fake", "model", "prompt")
        assert False, "expected ProviderError"
    except ProviderError:
        pass
