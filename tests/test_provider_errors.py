"""
Tests for the provider error-reporting path (providers.error_body).

Motivated by a real run: llama-server rejected a malformed GBNF
grammar with a 400 on every completion. Forge reported "400 Client
Error: Bad Request", the run finished in 16ms looking like a fast
success path, and the actual cause -- named explicitly in the
response body -- was only found by reading llama-server's log by
hand. These tests pin the body into the error message.
"""

import requests

from forge.errors import ProviderError
from forge.providers import error_body, llama_cpp, ollama, openrouter


class FakeResponse:
    def __init__(self, json_data=None, status_ok=True, text=""):
        self._json = json_data or {}
        self._status_ok = status_ok
        self.text = text

    def raise_for_status(self):
        if not self._status_ok:
            raise requests.HTTPError("400 Client Error: Bad Request")

    def json(self):
        return self._json


GRAMMAR_ERROR = (
    '{"error":{"message":"error parsing grammar: expecting newline or end at _call"}}'
)


# ---------------------------------------------------------------------
# error_body itself
# ---------------------------------------------------------------------


def test_error_body_returns_empty_for_blank_body():
    assert error_body(FakeResponse(text="")) == ""
    assert error_body(FakeResponse(text="   \n ")) == ""


def test_error_body_truncates_long_bodies():
    out = error_body(FakeResponse(text="x" * 5000), limit=100)
    assert len(out) < 300
    assert "5000 bytes total" in out


def test_error_body_never_raises_on_a_response_without_text():
    class NoText:
        @property
        def text(self):
            raise RuntimeError("body already consumed")

    assert error_body(NoText()) == ""


# ---------------------------------------------------------------------
# the three providers
# ---------------------------------------------------------------------


def _expect_error(fn):
    try:
        fn()
    except ProviderError as e:
        return str(e)
    raise AssertionError("expected ProviderError")


def test_llama_cpp_http_error_includes_the_backend_body(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse(status_ok=False, text=GRAMMAR_ERROR),
    )
    msg = _expect_error(lambda: llama_cpp.call("http://fake", "model", "prompt"))
    assert "error parsing grammar" in msg


def test_ollama_http_error_includes_the_backend_body(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse(
            status_ok=False, text='{"error":"model not found"}'
        ),
    )
    msg = _expect_error(lambda: ollama.call("http://fake", "model", "prompt"))
    assert "model not found" in msg


def test_openrouter_http_error_includes_the_backend_body(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **kw: FakeResponse(
            status_ok=False, text='{"error":"insufficient credits"}'
        ),
    )
    msg = _expect_error(
        lambda: openrouter.call("http://fake", "key", "model", "prompt")
    )
    assert "insufficient credits" in msg


def test_connection_failure_still_reports_transport_error_only(monkeypatch):
    """A request that never reached the server has no body to report --
    the message must stay clean rather than gain an empty suffix."""

    def boom(*a, **kw):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", boom)
    msg = _expect_error(lambda: llama_cpp.call("http://fake", "model", "prompt"))
    assert "connection refused" in msg
    assert "backend said" not in msg
