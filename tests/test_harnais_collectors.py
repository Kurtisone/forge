"""
Tests for the three MVP collectors, and for the distinction they exist
to preserve.

The assertion that matters in this file is the pair: a proxy that is
DOWN and a machine with NO containers running must not produce the
same Observation. graphs.sysadmin.running_containers() cannot tell them
apart -- its docstring says so -- and run #83fc443e is what a caller
does when handed that ambiguity. Every other test here is scaffolding
around that one.

Mocked at `_run_fixed`, sysadmin's single external boundary, which is
the same level tests/test_sysadmin.py mocks. The command CONSTRUCTION
is asserted separately (test_containers_asks_podman_through_the_proxy),
so patching the runner cannot hide a collector that talks to the host
socket directly.
"""

from datetime import datetime

import pytest

from forge import harnais
from forge.harnais import host_exec
from forge.harnais.collector import Observation
from forge.harnais.collectors import containers as containers_mod
from forge.harnais.collectors import cpu_ram as cpu_ram_mod
from forge.harnais.collectors import logs as logs_mod
from forge.harnais.collectors.containers import ContainersCollector
from forge.harnais.collectors.cpu_ram import CpuRamCollector
from forge.harnais.collectors.logs import KernelLogsCollector
from forge.harnais.facts import Fact

MEMINFO = """MemTotal:       15160368 kB
MemFree:         4903160 kB
MemAvailable:    9059480 kB
Buffers:           42080 kB
"""
LOADAVG = "0.14 0.39 0.68 1/1594 2759\n"


def _facts(observation: Observation) -> dict[str, object]:
    return {f"{f.domain}.{f.key}": f.value for f in observation.facts}


# --- the Observation invariant ---------------------------------------------


def test_an_observation_cannot_be_both_facts_and_failure():
    """A partial observation has no honest rendering, so it cannot be
    built -- see harnais/collector.py."""
    fact = Fact("ram", "used_pct", 40.2, "%", datetime.now(), "c")  # noqa: DTZ005
    with pytest.raises(ValueError, match="partial observation"):
        Observation("c", ("ram",), (fact,), "boom", datetime.now())  # noqa: DTZ005


def test_an_observation_must_say_something():
    """Success with nothing to show is indistinguishable from being
    unable to look, which is the whole bug."""
    with pytest.raises(ValueError, match="no facts and no error"):
        Observation("c", ("ram",), (), "", datetime.now())  # noqa: DTZ005


def test_an_observation_must_declare_its_domains():
    """A failure carries no facts, so nothing in it says what went
    unobserved unless the collector declared it first."""
    with pytest.raises(ValueError, match="must declare the domains"):
        Observation("c", (), (), "boom", datetime.now())  # noqa: DTZ005


# --- cpu_ram ----------------------------------------------------------------


def _patch_proc(monkeypatch, meminfo=MEMINFO, loadavg=LOADAVG):
    def fake_read(path):
        if path == cpu_ram_mod._MEMINFO:
            if meminfo is None:
                raise OSError("no such file")
            return meminfo
        if loadavg is None:
            raise OSError("no such file")
        return loadavg

    monkeypatch.setattr(cpu_ram_mod, "_read", fake_read)


def test_cpu_ram_reports_memory_and_load(monkeypatch):
    _patch_proc(monkeypatch)
    observation = CpuRamCollector().collect()

    assert not observation.failed
    assert _facts(observation) == {
        "ram.total_kb": 15160368,
        "ram.available_kb": 9059480,
        "ram.used_pct": 40.2,
        "cpu.load_1m": 0.14,
        "cpu.load_5m": 0.39,
        "cpu.load_15m": 0.68,
    }


def test_cpu_ram_names_proc_as_the_source(monkeypatch):
    """Every fact is traceable to the collector that read it, which is
    what makes `[fact] ... (cpu_ram, 17:40:12)` checkable."""
    _patch_proc(monkeypatch)
    assert {f.source for f in CpuRamCollector().collect().facts} == {"cpu_ram"}


@pytest.mark.parametrize(
    ("meminfo", "loadavg"),
    [(None, LOADAVG), (MEMINFO, None)],
)
def test_cpu_ram_fails_whole_when_either_file_fails(monkeypatch, meminfo, loadavg):
    """All or nothing: a half-read /proc would be a partially observed
    domain, and the three-way rule downstream has no room for one."""
    _patch_proc(monkeypatch, meminfo=meminfo, loadavg=loadavg)
    observation = CpuRamCollector().collect()

    assert observation.failed
    assert observation.facts == ()
    assert observation.domains == ("cpu", "ram")


def test_cpu_ram_fails_rather_than_guessing_a_missing_field(monkeypatch):
    """A /proc/meminfo without MemAvailable is not a /proc/meminfo.
    Substituting MemFree for it would be the exact move this package
    exists to prevent."""
    _patch_proc(monkeypatch, meminfo="MemTotal: 15160368 kB\nMemFree: 4903160 kB\n")
    observation = CpuRamCollector().collect()

    assert observation.failed
    assert "MemAvailable" in observation.error


# --- containers -------------------------------------------------------------


def _ps_line(name: str, started_at: int, status: str = "Up 6 minutes") -> str:
    """One row in the shape `podman ps --format` is asked for."""
    return f"{name}\t{started_at}\t{status}"


def test_containers_reports_what_is_running(monkeypatch):
    now = int(datetime.now().timestamp())  # noqa: DTZ005
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, timeout: "\n".join(
            [
                _ps_line("forge", now - 3600, "Up About an hour"),
                _ps_line("forge-llm", now - 247),
            ]
        ),
    )
    observation = ContainersCollector().collect()
    facts = _facts(observation)

    assert not observation.failed
    assert facts["container.running_count"] == 2
    assert facts["container.forge.status"] == "Up About an hour"
    assert facts["container.forge-llm.status"] == "Up 6 minutes"
    # Uptime is computed against the wall clock, so it is asserted as a
    # window rather than a value -- pinning the exact second would make
    # this test fail on a slow machine for a reason it does not check.
    assert 3595 <= facts["container.forge.uptime_s"] <= 3605
    assert 245 <= facts["container.forge-llm.uptime_s"] <= 255


def test_a_silent_restart_is_visible_without_any_history(monkeypatch):
    """
    The fault this collector was widened for. llama-server fell over
    twice on 2026-09-13 and `restart: unless-stopped` revived it with
    nothing saying so -- a series of measurements crossed a restart in
    silence.

    No history is needed to see it, and that matters because the World
    Model is in-memory and starts empty on every process: a single
    observation carries the uptime, so a question about the last hour
    meets a container that has been up four minutes.
    """
    now = int(datetime.now().timestamp())  # noqa: DTZ005
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, timeout: _ps_line("forge-llm", now - 247, "Up 4 minutes"),
    )
    facts = _facts(ContainersCollector().collect())

    assert facts["container.forge-llm.uptime_s"] < 300


def test_a_row_this_module_cannot_read_fails_loudly(monkeypatch):
    """
    `{{.Names}}` is the only one of the three fields this repo already
    runs in production. If podman renders either of the others
    differently, the observation must fail naming the line -- never
    report a container with a missing uptime, which reads exactly like
    one that has been up forever.
    """
    monkeypatch.setattr(
        containers_mod, "_run_fixed", lambda cmd, timeout: "forge-llm\tUp 6 minutes"
    )
    observation = ContainersCollector().collect()

    assert observation.failed
    assert observation.facts == ()
    assert "tab-separated" in observation.error
    assert "forge-llm" in observation.error


def test_a_non_numeric_started_at_fails_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, timeout: "forge-llm\t2026-09-14T09:12:03Z\tUp 6 minutes",
    )
    observation = ContainersCollector().collect()

    assert observation.failed
    assert "unix timestamp" in observation.error


def test_uptime_never_goes_negative_on_clock_skew():
    """podman reports the host's clock. A few seconds of skew must read
    as "just now", not as a container that starts in the future."""
    now = datetime.now()  # noqa: DTZ005
    assert containers_mod._uptime_s(int(now.timestamp()) + 30, now) == 0


def test_an_empty_podman_ps_is_an_observation_not_a_silence(monkeypatch):
    """
    Half of the pair. Nothing running is a FACT, and it stays sayable
    because the count exists even at zero.
    """
    monkeypatch.setattr(
        containers_mod, "_run_fixed", lambda cmd, timeout: "[no output]"
    )
    observation = ContainersCollector().collect()

    assert not observation.failed
    assert _facts(observation) == {"container.running_count": 0}


def test_a_dead_proxy_is_not_an_empty_machine(monkeypatch):
    """
    The other half, and the reason this package exists.

    Run #83fc443e: the podman proxy was down, the caller saw an empty
    list, and a model was handed the wrong evidence and a question
    about a named container. These two cases must never arrive at a
    reader looking the same.
    """
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, timeout: "[error] podman exited 125: unable to connect to socket",
    )
    down = ContainersCollector().collect()

    monkeypatch.setattr(
        containers_mod, "_run_fixed", lambda cmd, timeout: "[no output]"
    )
    empty = ContainersCollector().collect()

    assert down.failed and not empty.failed
    assert down.facts == ()
    assert "unable to connect" in down.error
    assert _facts(empty) == {"container.running_count": 0}


def test_no_output_never_becomes_a_container_named_no_output(monkeypatch):
    """The systemctl failure text once became two fake units called
    "System" and "Failed". Same class of mistake, different string."""
    monkeypatch.setattr(
        containers_mod, "_run_fixed", lambda cmd, timeout: "[no output]"
    )
    assert "container.[no output].status" not in _facts(ContainersCollector().collect())


def test_containers_asks_podman_through_the_proxy(monkeypatch):
    """
    Asserted on the command itself, not on the mocked runner: the
    collector must build its command with harnais.host_exec.podman_cmd,
    so SYSADMIN_PODMAN_URL still routes it at the read-only proxy
    rather than the host socket. Patching the flag on host_exec rather
    than on this module is the point -- it is what proves the wiring is
    read from the shared builder and not rebuilt here.
    """
    seen = {}

    def capture(cmd, timeout):
        seen["cmd"] = cmd
        return "[no output]"

    monkeypatch.setattr(containers_mod, "_run_fixed", capture)
    monkeypatch.setattr(host_exec, "SYSADMIN_PODMAN_URL", "tcp://127.0.0.1:9999")

    ContainersCollector().collect()

    assert seen["cmd"][:4] == ["podman", "--url", "tcp://127.0.0.1:9999", "ps"]
    # The format is asserted too: it is what keeps the reply one line
    # per container, under _run_fixed's line cap.
    assert seen["cmd"][-1] == containers_mod._PS_FORMAT


# --- logs -------------------------------------------------------------------


def test_kernel_logs_report_their_lines_and_their_text(monkeypatch):
    monkeypatch.setattr(
        logs_mod, "_run_fixed", lambda cmd, timeout: "line one\nline two"
    )
    observation = KernelLogsCollector().collect()

    assert _facts(observation) == {
        "logs.journalctl -k.lines": 2,
        "logs.journalctl -k.tail": "line one\nline two",
    }


def test_an_empty_journal_is_a_count_of_zero_not_a_missing_fact(monkeypatch):
    """An empty log block is not a quiet system: same reasoning as
    sysadmin's _nothing_collected_node, which exists because this model
    said an EMPTY block answered the question."""
    monkeypatch.setattr(logs_mod, "_run_fixed", lambda cmd, timeout: "[no output]")
    observation = KernelLogsCollector().collect()

    assert not observation.failed
    assert _facts(observation) == {"logs.journalctl -k.lines": 0}


def test_an_unreadable_journal_is_a_failure(monkeypatch):
    monkeypatch.setattr(
        logs_mod, "_run_fixed", lambda cmd, timeout: "[error] executable not found"
    )
    observation = KernelLogsCollector().collect()

    assert observation.failed
    assert observation.facts == ()


def test_the_logs_collector_has_no_target_parameter():
    """
    Deliberate, and load-bearing. A LogsCollector(unit=...) would be a
    path from router-chosen text to `journalctl --unit=<name>`, which
    is the one thing graphs/sysadmin.py spends its entire security
    model closing. Per-unit logs arrive with their validator, not
    before.
    """
    import inspect

    assert list(inspect.signature(KernelLogsCollector.collect).parameters) == ["self"]
    assert "unit" not in inspect.signature(KernelLogsCollector).parameters


# --- the harnais runner -----------------------------------------------------


def test_a_collector_that_raises_becomes_an_unobserved_domain(monkeypatch):
    """One broken collector must not take the other two with it, and a
    crash is just another way of not having observed something."""

    class Exploding:
        name = "boom"
        domains = ("gpu",)

        def is_available(self):
            return True

        def cost_hint(self):
            return "cheap"

        def collect(self):
            raise RuntimeError("segfault in the vendor blob")

    observations = harnais.observe([Exploding()])

    assert len(observations) == 1
    assert observations[0].failed
    assert "RuntimeError" in observations[0].error
    assert observations[0].domains == ("gpu",)


def test_an_unavailable_collector_is_reported_not_skipped():
    """
    Skipping it silently would put its domains back where this package
    started: absent from the context with nothing saying why, which
    reads as nothing to report.
    """

    class Absent:
        name = "gpu"
        domains = ("gpu",)

        def is_available(self):
            return False

        def cost_hint(self):
            return "cheap"

        def collect(self):  # pragma: no cover - must not be reached
            raise AssertionError("collect() called on an unavailable collector")

    observations = harnais.observe([Absent()])

    assert observations[0].failed
    assert "not available" in observations[0].error


def test_the_default_collectors_cover_the_mvp_domains():
    covered = {d for c in harnais.default_collectors() for d in c.domains}
    assert covered == {"cpu", "ram", "container", "logs"}


def test_a_container_list_on_the_line_cap_is_refused(monkeypatch):
    """
    `_run_fixed` truncates at SYSADMIN_MAX_LOG_LINES without saying so,
    so a reply sitting exactly on the cap may or may not be complete --
    and `running_count` would state the cap as an observed number. An
    under-count presented as a fact is the same fault as an empty list
    presented as an idle machine.
    """
    now = int(datetime.now().timestamp())  # noqa: DTZ005
    monkeypatch.setattr(containers_mod, "SYSADMIN_MAX_LOG_LINES", 3)
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, timeout: "\n".join(_ps_line(f"c{n}", now - 60) for n in range(3)),
    )
    observation = ContainersCollector().collect()

    assert observation.failed
    assert "truncated" in observation.error
    assert observation.facts == ()


def test_a_list_below_the_cap_is_reported_normally(monkeypatch):
    """The guard must not fire on an ordinary machine."""
    now = int(datetime.now().timestamp())  # noqa: DTZ005
    monkeypatch.setattr(containers_mod, "SYSADMIN_MAX_LOG_LINES", 3)
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, timeout: "\n".join(_ps_line(f"c{n}", now - 60) for n in range(2)),
    )
    observation = ContainersCollector().collect()

    assert not observation.failed
    assert _facts(observation)["container.running_count"] == 2
