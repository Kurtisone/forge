"""
Tests for forge.api: the HTTP surface stays open by default, and the
optional bearer-token auth (API_TOKEN) actually gates the endpoints
it's supposed to gate, and only those.

No network / no real LLM: call_llm is monkeypatched at the same
boundary used in test_orchestrator.py.
"""

import json

from fastapi.testclient import TestClient

import forge.api as api_mod
import forge.orchestrator as orch_mod


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
