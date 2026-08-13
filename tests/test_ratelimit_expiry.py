"""
Tests for the rate limiter's key lifecycle (security audit, M-2).

Counting was already correct; keeping was not. Every client IP that
ever made a request got a dict entry that lived as long as the
process. These tests are about what gets thrown away and, just as
importantly, what doesn't -- an over-eager expiry would hand a
throttled client a fresh allowance, which is worse than the leak it
replaced.
"""

import time

import pytest

from forge import ratelimit


@pytest.fixture(autouse=True)
def _clean_limiter(monkeypatch):
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_ENABLED", True)
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_keys_of_departed_clients_are_dropped(monkeypatch):
    """The leak itself: 50 one-shot clients used to leave 50 entries
    behind for good."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_REQUESTS", 5)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 0.05)

    for i in range(50):
        ratelimit.check(f"10.0.0.{i}")
    assert ratelimit.tracked_keys() == 50

    time.sleep(0.06)
    ratelimit.check("10.0.0.99")  # any request triggers the due sweep

    assert ratelimit.tracked_keys() == 1


def test_an_active_client_is_never_swept(monkeypatch):
    """Expiry must be observationally invisible. A client still inside
    its window keeps its counter -- otherwise the sweep is a way to
    reset the limit by waiting."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)

    for _ in range(3):
        assert ratelimit.check("192.168.1.5")[0] is True

    for i in range(20):  # traffic from elsewhere, sweeps may run
        ratelimit.check(f"172.16.0.{i}")

    allowed, retry_after = ratelimit.check("192.168.1.5")
    assert allowed is False
    assert retry_after >= 1


def test_a_returning_client_gets_the_full_allowance(monkeypatch):
    """The other half of the same guarantee: once a key has aged out,
    dropping it and keeping it are the same thing from the client's
    side."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 0.05)

    assert ratelimit.check("10.1.1.1")[0] is True
    assert ratelimit.check("10.1.1.1")[0] is True
    assert ratelimit.check("10.1.1.1")[0] is False

    time.sleep(0.06)
    assert ratelimit.check("10.1.1.1")[0] is True


def test_the_key_ceiling_bounds_a_burst_inside_one_window(monkeypatch):
    """The timed sweep bounds memory over time; it does nothing about
    a flood of distinct addresses arriving faster than one window. The
    ceiling is what covers that case."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_REQUESTS", 5)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 3600)
    monkeypatch.setattr(ratelimit, "_MAX_TRACKED_KEYS", 20)

    for i in range(200):
        ratelimit.check(f"10.2.{i // 256}.{i % 256}")

    assert ratelimit.tracked_keys() <= 21  # ceiling, plus the key just added


def test_reading_a_key_does_not_create_one(monkeypatch):
    """The dict was a defaultdict, so anything iterating or inspecting
    it by key could grow it -- including, eventually, the sweep meant
    to shrink it."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_REQUESTS", 5)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)

    ratelimit.check("10.3.3.3")
    before = ratelimit.tracked_keys()
    ratelimit._hits.get("never-seen")
    assert ratelimit.tracked_keys() == before


def test_disabled_limiter_stores_nothing(monkeypatch):
    """RATE_LIMIT_ENABLED=false should cost nothing at all, memory
    included -- it's the documented escape hatch for anyone fronting
    this with a proxy that already rate-limits."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_ENABLED", False)
    for i in range(100):
        assert ratelimit.check(f"10.4.0.{i}") == (True, 0)
    assert ratelimit.tracked_keys() == 0
