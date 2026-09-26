"""
/health says whether sysadmin can see the host, and never says anything
that would make a client think Forge is down.

The second half is the one that can hurt. The Android app chooses which
address to use by whether /health answers 2xx
(Forge_Android/.../ForgeRepository.kt, `firstReachable` reads
`isSuccessful` and nothing else), and decodes the body with Gson, which
ignores fields it does not know. So an ADDED field is invisible to it,
and a changed status code is not: a lost proxy reported as a 503 would
read, on the phone, as "Forge is unreachable" and send it hunting for
another address, when Forge is up and has only lost its view of the host.
"""

import datetime

import pytest
from fastapi.testclient import TestClient

import forge.api as api_mod
from forge import harnais, ratelimit
from forge.harnais import host_probe
from forge.harnais.collector import Observation

AT = datetime.datetime(2026, 9, 26, 12, 0)  # noqa: DTZ001


@pytest.fixture(autouse=True)
def _open_api(monkeypatch):
    ratelimit.reset()
    host_probe.reset()
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "FORGE_PROVIDER", "ollama")  # skip the llama probe
    yield
    ratelimit.reset()
    host_probe.reset()


def _health():
    return TestClient(api_mod.app).get("/health")


def _probe_says(monkeypatch, answer):
    monkeypatch.setattr(host_probe, "host_access", lambda: answer)


def test_health_carries_host_access_when_there_is_something_to_report(monkeypatch):
    _probe_says(monkeypatch, {"podman": "unobservable", "systemd": "ok"})
    body = _health().json()
    assert body["host_access"] == {"podman": "unobservable", "systemd": "ok"}


def test_health_is_exactly_what_it_was_when_there_is_nothing_to_report(monkeypatch):
    """Sysadmin off, or no proxy configured: not one new key, so a client
    that compares or displays the whole body sees no difference."""
    _probe_says(monkeypatch, {})
    assert set(_health().json()) == {"status", "provider", "model"}


def test_a_blind_host_is_still_a_200_and_still_status_ok(monkeypatch):
    """The Android invariant: Forge is up, so /health is 2xx and ok."""
    _probe_says(monkeypatch, {"podman": "unobservable", "systemd": "unobservable"})
    r = _health()
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_a_probe_that_raises_cannot_fail_health(monkeypatch):
    def boom():
        raise RuntimeError("the probe broke")

    monkeypatch.setattr(host_probe, "host_access", boom)
    r = _health()
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "host_access" not in r.json()


def test_the_real_probe_with_both_proxies_down_is_a_200(monkeypatch):
    """Through the real host_probe rather than a stub, so what is checked
    is the wiring and not only the handler."""
    monkeypatch.setattr(host_probe, "ENABLED_TOOLS", {"sysadmin"})
    monkeypatch.setattr(host_probe, "SYSADMIN_PODMAN_URL", "unix:///run/p/sock")
    monkeypatch.setattr(host_probe, "SYSADMIN_DBUS_ADDRESS", "unix:path=/run/d/bus")
    monkeypatch.setattr(
        harnais,
        "observe",
        lambda cs: [
            Observation.failure(c.name, c.domains, "[error] down", AT) for c in cs
        ],
    )

    r = _health()

    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["host_access"] == {
        "podman": "unobservable",
        "systemd": "unobservable",
    }
    host_probe._thread.join(5)


def test_an_unconfigured_deployment_never_runs_a_collector(monkeypatch):
    """The trap that cost four tests on CI: a collector the fixture did not
    patch reads the machine the suite runs on. With no proxy configured
    /health must not ask the Harnais anything at all."""
    monkeypatch.setattr(host_probe, "SYSADMIN_PODMAN_URL", "")
    monkeypatch.setattr(host_probe, "SYSADMIN_DBUS_ADDRESS", "")

    def forbidden(_collectors):
        raise AssertionError("a collector was run with no proxy configured")

    monkeypatch.setattr(harnais, "observe", forbidden)

    assert set(_health().json()) == {"status", "provider", "model"}
