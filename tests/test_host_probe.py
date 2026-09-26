"""
Whether Forge can see the host, reported where somebody already looks.

The 2026-09-11 outage left both host proxies in a restart loop for three
days and nothing said so, because the only place the blindness showed was
inside an answer to a question nobody asked. harnais/host_probe.py puts
it on /health. These tests fake the Harnais at its own boundary
(`harnais.observe`), never the machine: a probe that ran a real collector
here would answer for whatever the suite happens to run on, which is the
mistake test_the_blind_fixture_really_blinds_every_collector exists for.
"""

import datetime
import threading

import pytest

from forge import harnais
from forge.harnais import host_probe
from forge.harnais.collector import Observation
from forge.harnais.facts import Fact

AT = datetime.datetime(2026, 9, 26, 12, 0)  # noqa: DTZ001


def _ok(name, domain):
    return Observation.of(name, (domain,), [Fact(domain, "k", 1, None, AT, name)], AT)


def _down(name, domain, why="[error] podman exited 125: connection refused"):
    return Observation.failure(name, (domain,), why, AT)


class _Host:
    """A stand-in for harnais.observe: which collectors are 'down', an
    optional gate to hold a refresh open, and how often it was asked."""

    def __init__(self):
        self.calls = 0
        self.down: set[str] = set()
        self.boom: Exception | None = None
        self.gate: threading.Event | None = None

    def observe(self, collectors):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        if self.boom is not None:
            raise self.boom
        return [
            _down(c.name, c.domains[0])
            if c.name in self.down
            else _ok(c.name, c.domains[0])
            for c in collectors
        ]


class _Log:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(host_probe, "_now", lambda: now[0])
    return now


@pytest.fixture
def host(monkeypatch, clock):
    """Both proxies configured, sysadmin on, and a fake Harnais."""
    host_probe.reset()
    monkeypatch.setattr(host_probe, "ENABLED_TOOLS", {"sysadmin"})
    monkeypatch.setattr(host_probe, "SYSADMIN_PODMAN_URL", "unix:///run/p/sock")
    monkeypatch.setattr(host_probe, "SYSADMIN_DBUS_ADDRESS", "unix:path=/run/d/bus")
    fake = _Host()
    monkeypatch.setattr(harnais, "observe", fake.observe)
    yield fake
    if fake.gate is not None:
        fake.gate.set()
    if host_probe._thread is not None:
        host_probe._thread.join(5)
    host_probe.reset()


@pytest.fixture
def recorded(monkeypatch):
    rec = _Log()
    monkeypatch.setattr(host_probe, "log", rec)
    return rec


class TestWhenThereIsNothingToSay:
    def test_no_proxy_configured_means_no_probe(self, host, monkeypatch):
        monkeypatch.setattr(host_probe, "SYSADMIN_PODMAN_URL", "")
        monkeypatch.setattr(host_probe, "SYSADMIN_DBUS_ADDRESS", "")
        assert host_probe.host_access() == {}
        assert host.calls == 0

    def test_sysadmin_off_means_no_probe_even_with_proxies_configured(
        self, host, monkeypatch
    ):
        monkeypatch.setattr(host_probe, "ENABLED_TOOLS", {"chat", "code"})
        assert host_probe.host_access() == {}
        assert host.calls == 0

    def test_only_the_configured_proxy_is_reported(self, host, monkeypatch):
        monkeypatch.setattr(host_probe, "SYSADMIN_DBUS_ADDRESS", "")
        assert host_probe.host_access() == {"podman": "ok"}


class TestWhatItReports:
    def test_both_proxies_up(self, host):
        assert host_probe.host_access() == {"podman": "ok", "systemd": "ok"}

    def test_each_proxy_is_reported_on_its_own(self, host):
        host.down = {"units"}
        assert host_probe.host_access() == {"podman": "ok", "systemd": "unobservable"}

    def test_the_answer_is_only_ever_one_of_three_words(self, host):
        """No error text: /health is unauthenticated, and the reason
        carries socket paths. It goes to the log instead."""
        host.down = {"containers", "units"}
        got = host_probe.host_access()
        assert set(got.values()) <= {"ok", "unobservable", "unknown"}
        assert "sock" not in str(got) and "refused" not in str(got)


class TestCost:
    def test_the_answer_is_reused_inside_the_ttl(self, host, clock):
        host_probe.host_access()
        clock[0] += host_probe._TTL_S - 1
        host_probe.host_access()
        assert host.calls == 1

    def test_a_proxy_lost_after_the_ttl_shows_on_the_next_call(self, host, clock):
        assert host_probe.host_access()["podman"] == "ok"
        host.down = {"containers"}
        clock[0] += host_probe._TTL_S + 1
        assert host_probe.host_access()["podman"] == "unobservable"
        assert host.calls == 2

    def test_a_hung_proxy_cannot_hold_the_caller(self, host, monkeypatch):
        """The API serves everything from a pool of two workers. A probe
        stuck on a dead proxy has to cost a caller _WAIT_S at most."""
        monkeypatch.setattr(host_probe, "_WAIT_S", 0.05)
        host.gate = threading.Event()

        assert host_probe.host_access() == {"podman": "unknown", "systemd": "unknown"}

        host.gate.set()
        host_probe._thread.join(5)
        assert host_probe.host_access() == {"podman": "ok", "systemd": "ok"}

    def test_concurrent_callers_share_one_refresh(self, host, monkeypatch):
        monkeypatch.setattr(host_probe, "_WAIT_S", 5)
        host.gate = threading.Event()
        results: list[dict] = []

        def call():
            results.append(host_probe.host_access())

        callers = [threading.Thread(target=call) for _ in range(5)]
        for t in callers:
            t.start()
        # Let every caller reach the join before the refresh is allowed to end.
        threading.Event().wait(0.2)
        host.gate.set()
        for t in callers:
            t.join(5)

        assert host.calls == 1
        assert results == [{"podman": "ok", "systemd": "ok"}] * 5


class TestSayingItOnce:
    def test_losing_a_proxy_is_logged_once_not_on_every_refresh(
        self, host, clock, recorded
    ):
        host_probe.host_access()
        assert recorded.warnings == []

        host.down = {"containers"}
        for _ in range(3):
            clock[0] += host_probe._TTL_S + 1
            host_probe.host_access()

        assert len(recorded.warnings) == 1
        assert "podman" in recorded.warnings[0]
        assert "connection refused" in recorded.warnings[0]

    def test_getting_it_back_is_logged(self, host, clock, recorded):
        host.down = {"containers"}
        host_probe.host_access()
        host.down = set()
        clock[0] += host_probe._TTL_S + 1
        host_probe.host_access()

        assert len(recorded.infos) == 1
        assert "podman" in recorded.infos[0]

    def test_a_proxy_down_from_the_very_first_probe_is_logged(self, host, recorded):
        host.down = {"units"}
        host_probe.host_access()
        assert len(recorded.warnings) == 1
        assert "systemd" in recorded.warnings[0]


class TestWhenTheProbeItselfBreaks:
    def test_a_crash_leaves_the_last_answer_standing(self, host, clock, recorded):
        assert host_probe.host_access() == {"podman": "ok", "systemd": "ok"}
        host.boom = RuntimeError("boom")
        clock[0] += host_probe._TTL_S + 1

        assert host_probe.host_access() == {"podman": "ok", "systemd": "ok"}
        assert any("crashed" in w and "boom" in w for w in recorded.warnings)

    def test_a_crash_before_any_answer_is_unknown_not_an_exception(
        self, host, recorded
    ):
        host.boom = RuntimeError("boom")
        assert host_probe.host_access() == {"podman": "unknown", "systemd": "unknown"}
