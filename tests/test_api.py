"""
Tests for forge.api: the HTTP surface stays open by default, and the
optional bearer-token auth (API_TOKEN) actually gates the endpoints
it's supposed to gate, and only those.

No network / no real LLM: call_llm is monkeypatched at the same
boundary used in test_orchestrator.py.
"""

import json

import pytest
from fastapi.testclient import TestClient

import forge.api as api_mod
import forge.orchestrator as orch_mod
from forge import ratelimit


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # TestClient requests all share the same client key ("testclient"),
    # so counters would otherwise accumulate across every test in this
    # module regardless of which test made them.
    ratelimit.reset()
    yield
    ratelimit.reset()


def _client():
    return TestClient(api_mod.app)


def _mock_llm(monkeypatch, tool="chat", content="hi there"):
    monkeypatch.setattr(
        orch_mod, "call_llm", lambda prompt: json.dumps({"tool": tool, "content": content})
    )


# ── Open by default (API_TOKEN unset) ───────────────────────────────

def test_health_is_always_open(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    r = _client().get("/health")
    assert r.status_code == 200


def test_chat_open_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _mock_llm(monkeypatch)
    r = _client().post("/chat", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["output"] == "hi there"


def test_tools_open_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    r = _client().get("/tools")
    assert r.status_code == 200


# ── Gated when API_TOKEN is set ──────────────────────────────────────

def test_chat_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    _mock_llm(monkeypatch)
    r = _client().post("/chat", json={"message": "hello"})
    assert r.status_code == 401


def test_chat_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    _mock_llm(monkeypatch)
    r = _client().post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_chat_accepts_correct_token(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    _mock_llm(monkeypatch)
    r = _client().post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert r.status_code == 200
    assert r.json()["output"] == "hi there"


def test_health_stays_open_even_when_token_is_configured(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    r = _client().get("/health")
    assert r.status_code == 200


def test_traces_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    r = _client().get("/traces")
    assert r.status_code == 401


def test_run_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    r = _client().post("/run", json={"graph": "default", "input": "hi"})
    assert r.status_code == 401


# ── Rate limiting ─────────────────────────────────────────────────────

def test_requests_within_limit_all_succeed(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _mock_llm(monkeypatch)
    client = _client()
    for _ in range(5):
        r = client.post("/chat", json={"message": "hello"})
        assert r.status_code == 200


def test_exceeding_the_limit_returns_429(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)
    _mock_llm(monkeypatch)
    client = _client()

    for _ in range(3):
        assert client.post("/chat", json={"message": "hi"}).status_code == 200

    r = client.post("/chat", json={"message": "one too many"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_rate_limit_disabled_allows_unlimited_requests(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_ENABLED", False)
    _mock_llm(monkeypatch)
    client = _client()
    for _ in range(10):
        assert client.post("/chat", json={"message": "hi"}).status_code == 200


def test_health_is_never_rate_limited(monkeypatch):
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)
    client = _client()
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_different_clients_have_independent_limits(monkeypatch):
    """Sanity check on the limiter itself: two distinct keys must not
    share a counter (TestClient can't easily fake two source IPs, so
    this exercises forge.ratelimit.check directly)."""
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)

    allowed_a, _ = ratelimit.check("1.2.3.4")
    allowed_a_again, _ = ratelimit.check("1.2.3.4")
    allowed_b, _ = ratelimit.check("5.6.7.8")

    assert allowed_a is True
    assert allowed_a_again is False   # second hit from the same client, over the limit
    assert allowed_b is True          # different client, untouched by A's usage


def test_old_hits_expire_out_of_the_sliding_window(monkeypatch):
    """Once a hit ages past the window, it must stop counting against
    the limit -- the whole point of a sliding window over a hard
    counter that only resets in bulk."""
    import time

    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 0.05)

    allowed_first, _ = ratelimit.check("sliding-window-client")
    allowed_immediately_after, _ = ratelimit.check("sliding-window-client")
    time.sleep(0.06)
    allowed_after_expiry, _ = ratelimit.check("sliding-window-client")

    assert allowed_first is True
    assert allowed_immediately_after is False
    assert allowed_after_expiry is True
