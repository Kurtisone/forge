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
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": tool, "content": content}),
    )


# ── Open by default (API_TOKEN unset) ───────────────────────────────


def test_health_is_always_open(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "ollama")  # skip the llama_cpp probe
    r = _client().get("/health")
    assert r.status_code == 200


def test_health_reports_live_llama_cpp_model(monkeypatch):
    from forge.providers import llama_cpp

    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "llama_cpp")
    monkeypatch.setattr(api_mod, "LLM_MODEL", "configured-but-stale.gguf")
    monkeypatch.setattr(
        llama_cpp, "get_loaded_model", lambda url: "actually-loaded.gguf"
    )

    r = _client().get("/health")

    assert r.json()["model"] == "actually-loaded.gguf"


def test_health_falls_back_to_configured_model_when_probe_fails(monkeypatch):
    from forge.providers import llama_cpp

    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "llama_cpp")
    monkeypatch.setattr(api_mod, "LLM_MODEL", "configured.gguf")
    monkeypatch.setattr(llama_cpp, "get_loaded_model", lambda url: None)

    r = _client().get("/health")

    assert r.json()["model"] == "configured.gguf"


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


def test_tools_reports_what_the_policy_denies(monkeypatch):
    """
    /tools must agree with the router. A caller asking what Forge can
    do should not be told about a capability the router is no longer
    offered -- and a tool missing from the list should be explained
    rather than silently absent.
    """
    from forge.kernel import policy
    from forge.tools import registry as tool_registry

    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(
        tool_registry,
        "TOOLS",
        {"chat": lambda c: c, "web_search": lambda c: c},
    )

    body = _client().get("/tools").json()
    assert "web_search" in body["tools"]
    assert body["denied"] == []

    monkeypatch.setattr(policy, "POLICY_ALLOW_NETWORK", False)
    body = _client().get("/tools").json()
    assert "web_search" not in body["tools"]
    assert body["denied"] == ["web_search"]
    assert "chat" in body["tools"]


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
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "ollama")
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


def test_health_is_rate_limited_like_every_other_endpoint(monkeypatch):
    """Deliberately inverts the earlier test_health_is_never_rate_limited
    (audit M-3).

    Exempting /health looked free when it was read as "a status string
    nobody can abuse". It isn't one: with FORGE_PROVIDER=llama_cpp,
    every hit makes an outbound HTTP call to the inference server to
    read the loaded model name. Unmetered, that turns one cheap
    anonymous request into load on the LLM -- an amplifier reachable
    without a token, which is the combination that matters.

    Still unauthenticated (see the test below) -- metered, not gated.
    """
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(api_mod.ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "ollama")
    client = _client()

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200

    r = client.get("/health")
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_health_stays_unauthenticated_even_with_a_token_set(monkeypatch):
    """The rate limit must not have quietly turned into an auth gate:
    a container healthcheck and the UI's own status line both call
    /health before they have a token to send."""
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "ollama")
    assert _client().get("/health").status_code == 200


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
    assert allowed_a_again is False  # second hit from the same client, over the limit
    assert allowed_b is True  # different client, untouched by A's usage


# ── Vector memory / RAG (v3.7) ───────────────────────────────────────


def _mock_embed(monkeypatch, vec=None):
    from forge import rag

    monkeypatch.setattr(rag, "_embed", lambda text: vec or [0.1] * rag.EMBEDDING_DIM)


def _use_tmp_rag_db(tmp_path, monkeypatch):
    from forge import rag

    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))


def test_remember_open_when_no_token_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    r = _client().post(
        "/remember",
        json={"kind": "decision", "content": "use sqlite-vec", "project": "forge"},
    )
    assert r.status_code == 200
    assert r.json()["id"] == 1


def test_remember_requires_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    r = _client().post(
        "/remember", json={"kind": "decision", "content": "x", "project": None}
    )
    assert r.status_code == 401


def test_remember_rejects_empty_content(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    r = _client().post("/remember", json={"kind": "todo", "content": "   "})
    assert r.status_code == 400


def test_remember_accepts_fact_kind(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    r = _client().post(
        "/remember", json={"kind": "fact", "content": "Possède un Steam Deck"}
    )
    assert r.status_code == 200


def test_remember_rejects_invalid_kind(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    r = _client().post("/remember", json={"kind": "note", "content": "x"})
    assert r.status_code == 422  # pydantic Literal validation


def test_remember_returns_502_when_embedding_server_down(monkeypatch, tmp_path):
    from forge import rag

    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_rag_db(tmp_path, monkeypatch)

    def _raise(text):
        raise rag.EmbeddingError("connection refused")

    monkeypatch.setattr(rag, "_embed", _raise)

    r = _client().post("/remember", json={"kind": "decision", "content": "x"})
    assert r.status_code == 502


def test_search_returns_stored_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    _client().post(
        "/remember",
        json={"kind": "decision", "content": "use sqlite-vec", "project": "forge"},
    )

    r = _client().get("/search", params={"q": "sqlite"})
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["content"] == "use sqlite-vec"


def test_search_filters_by_kind_and_project(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    client = _client()
    client.post(
        "/remember",
        json={"kind": "decision", "content": "forge decision", "project": "forge"},
    )
    client.post(
        "/remember",
        json={"kind": "todo", "content": "nipogi todo", "project": "nipogi"},
    )

    r = client.get("/search", params={"q": "anything", "kind": "todo"})
    assert len(r.json()) == 1
    assert r.json()[0]["project"] == "nipogi"


def test_search_requires_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    _use_tmp_rag_db(tmp_path, monkeypatch)
    _mock_embed(monkeypatch)

    r = _client().get("/search", params={"q": "anything"})
    assert r.status_code == 401


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


# ── Drawer / compaction (v3.9) ──────────────────────────────────────


def _use_tmp_memory(tmp_path, monkeypatch):
    from forge import memory

    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))


def test_history_reflects_persisted_messages(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_memory(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, content="hi there")

    client = _client()
    client.post("/chat", json={"message": "hello"})

    r = client.get("/history")
    assert r.status_code == 200
    body = r.json()
    assert [m["role"] for m in body] == ["user", "assistant"]
    assert all(m["pinned"] is False for m in body)


def test_pin_then_appears_in_drawer(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_memory(tmp_path, monkeypatch)
    _mock_llm(monkeypatch, content="hi there")

    client = _client()
    client.post("/chat", json={"message": "hello"})
    message_id = client.get("/history").json()[0]["id"]

    r = client.post("/drawer/pin", json={"message_id": message_id})
    assert r.status_code == 200

    drawer = client.get("/drawer").json()
    assert [m["id"] for m in drawer] == [message_id]

    r = client.post("/drawer/unpin", json={"message_id": message_id})
    assert r.status_code == 200
    assert client.get("/drawer").json() == []


def test_pin_unknown_id_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_memory(tmp_path, monkeypatch)

    r = _client().post("/drawer/pin", json={"message_id": 9999})
    assert r.status_code == 404


def test_drawer_requires_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    _use_tmp_memory(tmp_path, monkeypatch)

    r = _client().get("/drawer")
    assert r.status_code == 401


def test_manual_compact_reports_removed_count(monkeypatch, tmp_path):
    from forge import compaction

    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    _use_tmp_memory(tmp_path, monkeypatch)
    monkeypatch.setattr(compaction, "COMPACTION_KEEP_RECENT", 1)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(compaction.rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(
        compaction.rag, "remember", lambda conn, kind, content, project: 1
    )

    client = _client()
    _mock_llm(monkeypatch, content="hi there")
    for i in range(3):
        client.post("/chat", json={"message": f"msg{i}"})

    r = client.post("/compact")
    assert r.status_code == 200
    assert r.json()["removed"] > 0


def test_jobs_endpoint_lists_delegation_jobs():
    """
    Forge's own interface is the thread (zero tabs), so this endpoint
    is not where a job gets read day to day -- it exists so "what is
    that job doing" has an answer that isn't cat data/jobs.json over
    SSH from a phone.
    """
    from forge import jobs

    job = jobs.create({"objective": "réparer le cache"})
    jobs.transition(job.id, jobs.AWAITING_USER, pending_field="workspace")

    response = _client().get("/jobs")
    assert response.status_code == 200
    listed = response.json()["jobs"]
    assert [j["id"] for j in listed] == [job.id]
    assert listed[0]["status"] == jobs.AWAITING_USER
    assert listed[0]["pending_field"] == "workspace"


def test_startup_reconciles_jobs_left_running(monkeypatch):
    """
    The reconciliation only helps if it runs on the path a restart
    actually takes. Wiring it into lifespan and never checking that
    lifespan calls it would leave the lie on disk untouched in exactly
    the case it was written for.
    """
    import asyncio

    from forge import jobs

    job = jobs.create()
    jobs.transition(job.id, jobs.READY)
    jobs.transition(job.id, jobs.RUNNING)

    monkeypatch.setattr(api_mod, "check_auth_configuration", lambda: None)

    async def _startup():
        async with api_mod.lifespan(api_mod.app):
            pass

    asyncio.run(_startup())
    assert jobs.get(job.id).status == jobs.INTERRUPTED
