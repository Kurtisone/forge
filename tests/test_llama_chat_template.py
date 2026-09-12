"""
Forge asks the served model what framing it wants, instead of knowing.

A raw prompt to /completion applies no chat template, so the model sees
plain text where its training saw role markers. That is invisible while
one model is served -- Qwen3.5-9B, which Forge was tuned around, handles
it -- and it is the difference between four malformed replies in
thirty-six and zero on a model whose template is ChatML (measured
2026-09-12, bench/lfm-settings-sweep.json).

The framing is not written down anywhere in Forge. llama-server's
/apply-template renders the LOADED model's own template, so a sentinel
sent through it comes back with the answer on either side. Nothing here
knows whether that is ChatML, Llama-3, or a format that does not exist
yet.
"""

import pytest
import requests

import forge.providers.llama_cpp as provider


@pytest.fixture(autouse=True)
def _fresh_cache():
    provider.reset_template_cache()
    yield
    provider.reset_template_cache()


class _Resp:
    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc

    def raise_for_status(self):
        pass

    def json(self):
        if self._exc:
            raise self._exc
        return self._payload


def _server(monkeypatch, payload=None, exc=None, counter=None):
    def post(url, **kwargs):
        if counter is not None:
            counter.append(url)
        if exc and not payload:
            raise exc
        return _Resp(payload, exc if payload is None else None)

    monkeypatch.setattr(provider.requests, "post", post)


CHATML = (
    "<|im_start|>user\n" + provider._SENTINEL + "<|im_end|>\n<|im_start|>assistant\n"
)


def test_the_framing_comes_from_the_server(monkeypatch):
    _server(monkeypatch, {"prompt": CHATML})
    prefix, suffix = provider._framing("http://x")
    assert prefix == "<|im_start|>user\n"
    assert suffix == "<|im_end|>\n<|im_start|>assistant\n"


def test_a_format_nobody_wrote_down_works_the_same(monkeypatch):
    """The point of asking rather than knowing."""
    _server(monkeypatch, {"prompt": f"[[[START]]]{provider._SENTINEL}[[[GO]]]"})
    assert provider._framing("http://x") == ("[[[START]]]", "[[[GO]]]")


def test_it_is_asked_once_and_then_remembered(monkeypatch):
    calls = []
    _server(monkeypatch, {"prompt": CHATML}, counter=calls)
    for _ in range(5):
        provider._framing("http://x")
    assert len(calls) == 1, (
        "an HTTP round-trip in front of every routing decision is not a "
        "wrapper, it is a latency regression"
    )


class TestItDegradesToTheRawPrompt:
    """
    Every failure here means "send what Forge sent before this
    existed". A wrapper is not worth failing a turn over.
    """

    def test_an_older_llama_cpp_with_no_such_endpoint(self, monkeypatch):
        _server(monkeypatch, exc=requests.RequestException("404"))
        assert provider._framing("http://x") == ("", "")

    def test_a_response_that_is_not_json(self, monkeypatch):
        _server(monkeypatch, payload={}, exc=ValueError("nope"))
        assert provider._framing("http://x") == ("", "")

    def test_a_template_that_swallowed_the_sentinel(self, monkeypatch):
        """
        Splitting on something that is not there gives ("", everything),
        which would prepend nothing and append the whole template.
        """
        _server(monkeypatch, {"prompt": "<|im_start|>user\n<|im_end|>"})
        assert provider._framing("http://x") == ("", "")

    def test_a_template_that_repeats_the_sentinel(self, monkeypatch):
        """
        A system preamble echoing the message would put half the
        framing on the wrong side of the split.
        """
        s = provider._SENTINEL
        _server(monkeypatch, {"prompt": f"sys: {s}\nuser: {s}\nassistant:"})
        assert provider._framing("http://x") == ("", "")

    def test_a_prompt_field_of_the_wrong_type(self, monkeypatch):
        _server(monkeypatch, {"prompt": None})
        assert provider._framing("http://x") == ("", "")


class TestTheKnobDecides:
    def test_off_sends_the_prompt_untouched(self, monkeypatch):
        monkeypatch.setattr(provider, "LLAMA_CPP_APPLY_TEMPLATE", False)
        sent = {}

        def post(url, json=None, **kw):
            sent["prompt"] = json["prompt"]
            return _Resp({"content": '{"tool":"chat","content":"ok"}'})

        monkeypatch.setattr(provider.requests, "post", post)
        provider.call("http://x", "m", "PROMPT")
        assert sent["prompt"] == "PROMPT"

    def test_on_wraps_it(self, monkeypatch):
        monkeypatch.setattr(provider, "LLAMA_CPP_APPLY_TEMPLATE", True)
        sent = {}

        def post(url, json=None, **kw):
            if url.endswith("/apply-template"):
                return _Resp({"prompt": CHATML})
            sent["prompt"] = json["prompt"]
            return _Resp({"content": '{"tool":"chat","content":"ok"}'})

        monkeypatch.setattr(provider.requests, "post", post)
        provider.call("http://x", "m", "PROMPT")
        assert sent["prompt"] == (
            "<|im_start|>user\nPROMPT<|im_end|>\n<|im_start|>assistant\n"
        )

    def test_it_ships_off(self):
        """
        The house rule: a default is measured, not chosen. On a
        single-model deployment this buys nothing measured and costs
        the pure-append property v3.12 bought.
        """
        import forge.config as cfg

        assert cfg.LLAMA_CPP_APPLY_TEMPLATE is False
