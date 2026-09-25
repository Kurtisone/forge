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
    monkeypatch.setattr(api_mod, "API_TOKEN", "a" * 48)
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    api_mod.check_auth_configuration()  # must not raise


def test_allows_a_token_exactly_at_the_floor(monkeypatch):
    monkeypatch.setattr(api_mod, "API_TOKEN", "a" * api_mod.MIN_API_TOKEN_LENGTH)
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    api_mod.check_auth_configuration()  # must not raise


@pytest.mark.parametrize(
    "token",
    [
        "1234",  # four digits: the shape a chosen token takes
        "s3cret",
        "a" * (api_mod.MIN_API_TOKEN_LENGTH - 1),  # one short of the floor
    ],
)
def test_refuses_a_token_shorter_than_the_floor(monkeypatch, token):
    monkeypatch.setattr(api_mod, "API_TOKEN", token)
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    with pytest.raises(api_mod.InsecureConfiguration):
        api_mod.check_auth_configuration()


def test_a_short_token_is_refused_even_when_unauthenticated_is_allowed(monkeypatch):
    """
    API_ALLOW_UNAUTHENTICATED is the way to run with NO token. With a
    token set, require_token enforces it whatever that flag says, so
    the flag cannot make a short one acceptable.
    """
    monkeypatch.setattr(api_mod, "API_TOKEN", "1234")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", True)
    with pytest.raises(api_mod.InsecureConfiguration):
        api_mod.check_auth_configuration()


def test_the_short_token_refusal_says_the_length_and_never_the_value(monkeypatch):
    """
    The message goes to the container log, which outlives the process
    and is read by more than the person who set the token. It says what
    is wrong and how to fix it, and it does not repeat the secret.
    """
    token = "hunter2-hunter2"
    monkeypatch.setattr(api_mod, "API_TOKEN", token)
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    with pytest.raises(api_mod.InsecureConfiguration) as excinfo:
        api_mod.check_auth_configuration()
    message = str(excinfo.value)
    assert token not in message
    assert str(len(token)) in message
    assert str(api_mod.MIN_API_TOKEN_LENGTH) in message
    assert "token_hex" in message


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


@pytest.mark.parametrize("token", ["", "1234"])
def test_lifespan_actually_runs_the_check(monkeypatch, token):
    monkeypatch.setattr(api_mod, "API_TOKEN", token)
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", False)
    with pytest.raises(api_mod.InsecureConfiguration), TestClient(api_mod.app):
        pass  # pragma: no cover -- entering the context is what raises
