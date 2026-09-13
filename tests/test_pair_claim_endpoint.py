"""
Tests for POST /pair/claim: the one open endpoint that hands out the
bearer token.

Open because it has to be -- a device claiming its first credential
has none to present. So everything that would normally be the auth
check's job is a property of the token instead, and each of those is
pinned here: single use, expiry, and a refusal that says the same
thing whatever went wrong.
"""

import pytest
from fastapi.testclient import TestClient

import forge.api as api_mod
from forge import pairing, ratelimit

_BEARER = "durable-bearer-token-that-must-not-leak"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # TestClient requests all share one client key, so counters would
    # accumulate across this module's tests.
    ratelimit.reset()
    yield
    ratelimit.reset()


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", "http://10.8.0.1:8000")
    monkeypatch.setattr(pairing, "API_TOKEN", _BEARER)
    pairing.reset()
    yield
    pairing.reset()


def _client():
    return TestClient(api_mod.app)


def test_a_fresh_token_is_exchanged_for_the_bearer_token():
    r = _client().post("/pair/claim", json={"token": pairing.issue()})

    assert r.status_code == 200
    assert r.json()["token"] == _BEARER


def test_no_bearer_token_is_required_to_claim(monkeypatch):
    """
    The point of the endpoint. A device pairing for the first time has
    nothing to authenticate with, so requiring a token here would make
    pairing impossible.
    """
    monkeypatch.setattr(api_mod, "API_TOKEN", "server-side-token-is-set")
    r = _client().post("/pair/claim", json={"token": pairing.issue()})

    assert r.status_code == 200


def test_the_same_token_cannot_be_claimed_twice():
    token = pairing.issue()
    client = _client()

    assert client.post("/pair/claim", json={"token": token}).status_code == 200
    assert client.post("/pair/claim", json={"token": token}).status_code == 401


def test_an_expired_token_is_refused(monkeypatch):
    import time

    monkeypatch.setattr(pairing, "PAIRING_TTL_SECONDS", 0.05)
    token = pairing.issue()
    time.sleep(0.1)

    assert _client().post("/pair/claim", json={"token": token}).status_code == 401


@pytest.mark.parametrize("token", ["", "   ", "not-a-real-token"])
def test_a_token_that_was_never_issued_is_refused(token):
    assert _client().post("/pair/claim", json={"token": token}).status_code == 401


def test_every_refusal_reads_the_same():
    """
    Unknown, already claimed and expired answer identically. Naming
    which one it was would confirm to a prober that a token existed,
    and "expired" is a different sentence from "never existed".
    """
    used = pairing.issue()
    client = _client()
    client.post("/pair/claim", json={"token": used})

    bodies = {
        client.post("/pair/claim", json={"token": t}).json()["detail"]
        for t in (used, "never-issued")
    }
    assert len(bodies) == 1


def test_a_refused_claim_leaks_nothing():
    r = _client().post("/pair/claim", json={"token": "not-a-real-token"})
    assert _BEARER not in r.text


def test_claiming_is_rate_limited(monkeypatch):
    """
    The only control standing between a prober and a 256-bit guess.
    Without it, an open endpoint that returns the bearer token on a
    match is an unmetered oracle.
    """
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)
    ratelimit.reset()
    client = _client()

    for _ in range(3):
        assert client.post("/pair/claim", json={"token": "wrong"}).status_code == 401

    r = client.post("/pair/claim", json={"token": "wrong"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_pair_claim_is_the_only_endpoint_this_opened():
    """
    Forcing function, not a description. /pair/claim is the third
    route reachable without a bearer token, and the first one added
    since the app started refusing to boot open. An endpoint that
    forgets Depends(require_token) is indistinguishable from one that
    omits it on purpose -- from the diff, from the docstring, and from
    every other test in this suite. So the set is pinned: adding to it
    has to be a deliberate edit here.
    """
    from fastapi.routing import APIRoute

    open_routes = set()
    for route in api_mod.app.routes:
        if not isinstance(route, APIRoute):
            continue
        guards = {d.call.__name__ for d in route.dependant.dependencies if d.call}
        if "require_token" not in guards:
            open_routes.add(route.path)

    assert open_routes == {"/", "/health", "/pair/claim"}
