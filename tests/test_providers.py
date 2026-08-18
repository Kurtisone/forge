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
    assert llama_cpp.call("http://fake", "model", "prompt").text == "hello"


def test_llama_cpp_call_accepts_completion_key(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **kw: FakeResponse({"completion": "hi"})
    )
    assert llama_cpp.call("http://fake", "model", "prompt").text == "hi"


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


def test_llama_cpp_sends_fixed_id_slot_and_cache_prompt_by_default(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"content": "hi"})

    monkeypatch.setattr(requests, "post", fake_post)
    llama_cpp.call("http://fake", "model", "prompt")

    assert captured["payload"]["id_slot"] == 0
    assert captured["payload"]["cache_prompt"] is True


def test_llama_cpp_id_slot_configurable(monkeypatch):
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_ID_SLOT", 3)
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"content": "hi"})

    monkeypatch.setattr(requests, "post", fake_post)
    llama_cpp.call("http://fake", "model", "prompt")

    assert captured["payload"]["id_slot"] == 3


def test_llama_cpp_cache_prompt_disabled_via_config(monkeypatch):
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_CACHE_PROMPT", False)
    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return FakeResponse({"content": "hi"})

    monkeypatch.setattr(requests, "post", fake_post)
    llama_cpp.call("http://fake", "model", "prompt")

    assert captured["payload"]["cache_prompt"] is False


def test_llama_cpp_logs_ms_per_token(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse(
            {
                "content": "hi",
                "tokens_evaluated": 100,
                "tokens_cached": 12,
                "timings": {"prompt_ms": 250.0},
            }
        ),
    )
    events = []
    monkeypatch.setattr(
        llama_cpp.log, "event", lambda name, **fields: events.append((name, fields))
    )

    llama_cpp.call("http://fake", "model", "prompt")

    assert events == [
        (
            "llama_cpp.cache",
            {
                "prompt_n": 100,
                "prompt_ms": 250.0,
                "ms_per_token": 2.5,
                "tokens_cached": 12,
            },
        )
    ]


def test_llama_cpp_falls_back_to_timings_prompt_n(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse(
            {"content": "hi", "timings": {"prompt_n": 50, "prompt_ms": 100.0}}
        ),
    )
    events = []
    monkeypatch.setattr(
        llama_cpp.log, "event", lambda name, **fields: events.append((name, fields))
    )

    llama_cpp.call("http://fake", "model", "prompt")

    assert events == [
        (
            "llama_cpp.cache",
            {
                "prompt_n": 50,
                "prompt_ms": 100.0,
                "ms_per_token": 2.0,
                "tokens_cached": None,
            },
        )
    ]


def test_llama_cpp_ms_per_token_none_when_prompt_ms_missing(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse({"content": "hi", "tokens_evaluated": 100}),
    )
    events = []
    monkeypatch.setattr(
        llama_cpp.log, "event", lambda name, **fields: events.append((name, fields))
    )

    llama_cpp.call("http://fake", "model", "prompt")

    assert events == [
        (
            "llama_cpp.cache",
            {
                "prompt_n": 100,
                "prompt_ms": None,
                "ms_per_token": None,
                "tokens_cached": None,
            },
        )
    ]


def test_llama_cpp_no_cache_log_when_prompt_n_missing(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **kw: FakeResponse({"content": "hi"})
    )
    events = []
    monkeypatch.setattr(
        llama_cpp.log, "event", lambda name, **fields: events.append((name, fields))
    )

    llama_cpp.call("http://fake", "model", "prompt")

    assert events == []


def test_get_loaded_model_reads_model_path(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: FakeResponse({"model_path": "/models/qwen3-4b.gguf"}),
    )
    assert llama_cpp.get_loaded_model("http://fake") == "qwen3-4b.gguf"


def test_get_loaded_model_falls_back_to_nested_field(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: FakeResponse(
            {"default_generation_settings": {"model": "/models/other.gguf"}}
        ),
    )
    assert llama_cpp.get_loaded_model("http://fake") == "other.gguf"


def test_get_loaded_model_returns_none_on_network_failure(monkeypatch):
    def _raise(*a, **kw):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(requests, "get", _raise)
    assert llama_cpp.get_loaded_model("http://fake") is None


def test_get_loaded_model_returns_none_when_no_known_field(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse({"foo": "bar"}))
    assert llama_cpp.get_loaded_model("http://fake") is None


def test_get_loaded_model_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **kw: FakeResponse({}, status_ok=False)
    )
    assert llama_cpp.get_loaded_model("http://fake") is None


# ---------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------


def test_ollama_call_returns_response_field(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **kw: FakeResponse({"response": "hello"})
    )
    assert ollama.call("http://fake", "model", "prompt").text == "hello"


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
        lambda *a, **kw: FakeResponse({"choices": [{"message": {"content": "hello"}}]}),
    )
    assert openrouter.call("http://fake", "sk-fake", "model", "prompt").text == "hello"


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


def test_grammar_selection_prefers_the_callers_grammar(monkeypatch):
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_USE_GRAMMAR", True)
    assert llama_cpp._grammar_for('root ::= "x"') == 'root ::= "x"'


def test_grammar_selection_falls_back_to_the_router_grammar(monkeypatch):
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_USE_GRAMMAR", True)
    grammar = llama_cpp._grammar_for(None)
    assert grammar is not None
    assert "root" in grammar


def test_grammar_knob_off_beats_an_explicit_grammar(monkeypatch):
    """
    LLAMA_CPP_USE_GRAMMAR exists to take grammar sampling out of the
    picture while debugging. A knob with an exception isn't one.
    """
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_USE_GRAMMAR", False)
    assert llama_cpp._grammar_for('root ::= "x"') is None


def test_an_invalid_grammar_is_dropped_rather_than_sent(monkeypatch, caplog):
    """
    llama-server answers 400 to every completion whose grammar it
    can't parse, so sending one is a guaranteed TOTAL outage (v3.10:
    the router didn't degrade, it died). Running unconstrained falls
    back to the parser's existing chain instead -- and the log line
    names the offending rule, which the server's 400 body never does.
    """
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_USE_GRAMMAR", True)
    with caplog.at_level("ERROR"):
        assert llama_cpp._grammar_for('root ::= spec_call\nspec_call ::= "x"') is None
    assert "spec_call" in caplog.text
