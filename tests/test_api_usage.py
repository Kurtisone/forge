"""
Getting the run's token counts out to a client.

The interesting part is not the plumbing, it is WHERE the snapshot is
taken. metrics keeps its totals in a contextvar so concurrent requests
cannot sum into each other, and api.chat() dispatches run() through a
worker thread. A snapshot taken after run() returns therefore sees
either nothing or another request's scope -- which is the bug these
tests exist to keep out, since it would fail silently and intermittently
rather than loudly.
"""

import json

import pytest
from fastapi.testclient import TestClient

import forge.api as api_mod
import forge.orchestrator as orch_mod
from forge.api import app
from forge.tools.registry import TOOLS


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", True)
    # No llama-server in tests; the limit is then simply unknown.
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "ollama")
    return TestClient(app)


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setitem(TOOLS, "chat", lambda content: content)
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda p: json.dumps({"tool": "chat", "content": "ok", "done": True}),
    )


def test_the_snapshot_survives_the_thread_hop(client, scripted):
    """The whole point. None here would mean the snapshot is being
    taken outside the run scope, which is the failure this guards.

    llm_calls is 0, not missing: the fixture patches
    orchestrator.call_llm, so llm.call_llm -- and with it
    metrics.record -- never runs. That makes this the clean
    demonstration of the convention: an OPEN scope that recorded
    nothing reports zeros, while no scope at all reports None. The two
    are different facts and the gauge treats them differently."""
    r = client.post("/chat", json={"message": "bonjour"})
    assert r.status_code == 200
    usage = r.json()["usage"]
    assert usage is not None
    assert usage["llm_calls"] == 0
    assert "prompt_tokens" in usage


def test_chat_usage_is_marked_as_measured(client, scripted):
    """Counts from a real call are not estimates, and the client has to
    be able to tell -- the gauge renders them differently."""
    r = client.post("/chat", json={"message": "bonjour"})
    assert r.json()["usage"]["estimated"] is False


def test_context_is_marked_as_estimated(client):
    """The exact count cannot exist before the prompt has been sent."""
    r = client.get("/context")
    assert r.status_code == 200
    assert r.json()["estimated"] is True


def test_context_reports_a_cost_with_no_history(client):
    """A fresh conversation is not free: the static template is most of
    the prompt. A gauge starting at zero would be lying."""
    body = client.get("/context").json()
    assert body["prompt_tokens"] > 0


def test_an_unknown_context_limit_is_none_not_a_guess(client):
    """Providers that cannot report a window get no denominator rather
    than a plausible-looking constant -- a gauge against a number
    nobody maintains is worse than a gauge with no denominator."""
    assert client.get("/context").json()["context_limit"] is None


def test_result_usage_is_none_outside_a_run():
    """AgentState built directly, no run(), no accounting scope. None
    rather than zeros: "not reported" and "reported as nothing" are
    different facts."""
    from forge.types import AgentState

    state = AgentState(user_input="x", max_steps=1)
    assert state.to_result().usage is None
