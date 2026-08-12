"""
Tests for the startup auth guard (security audit, C-1).

The guard runs in the app's lifespan, so it fires on a real uvicorn
boot and on TestClient used as a context manager -- not on a bare
TestClient(app), which is why the rest of the suite (which builds
clients that way, with API_TOKEN monkeypatched per-test) is
unaffected.
"""

import pytest
from fastapi.testclient import TestClient

import forge.api as api_mod


def test_refuses_when_token_missing_and_not_opted_out(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    with pytest.raises(api_mod.InsecureConfiguration):
        api_mod.check_auth_configuration()


def test_allows_when_token_is_set(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "s3cret")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    api_mod.check_auth_configuration()  # must not raise


def test_allows_when_explicitly_opted_out(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", True)
    api_mod.check_auth_configuration()  # must not raise


def test_error_message_names_both_ways_out(monkeypatch):
    """
    A refusal that doesn't say how to proceed just gets worked around
    by whatever the first search result suggests.
    """
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    with pytest.raises(api_mod.InsecureConfiguration) as excinfo:
        api_mod.check_auth_configuration()
    message = str(excinfo.value)
    assert "API_TOKEN" in message
    assert "API_ALLOW_UNAUTHENTICATED" in message


def test_lifespan_actually_runs_the_check(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    with pytest.raises(api_mod.InsecureConfiguration), TestClient(api_mod.app):
        pass  # pragma: no cover -- entering the context is what raises
