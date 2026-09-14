"""
The shared host plumbing: stated once, and the layering that says so.

forge/harnais/host_exec.py exists because three modules run commands
against this machine and only one of them used to own the code for it.
The other two imported private names out of graphs/sysadmin.py, which
the collectors' docstrings defended as the lesser evil: rebuilding
`podman ps` elsewhere would duplicate the proxy wiring, the minimal
subprocess env, the timeout and the "[error] " convention.

These tests pin what the move bought. They are not a re-test of
run_fixed's behaviour -- tests/test_sysadmin.py has covered that since
2026-08-11 and still does, through the names the graph re-exports,
which also proves those aliases are live.
"""

import pytest

import forge.graphs.sysadmin as sysadmin_mod
from forge.harnais import host_exec
from forge.harnais.collectors import containers as containers_mod

_URL = "tcp://127.0.0.1:9999"
_DIR = "/host-journal"


class TestTheProxyWiringIsStatedOnce:
    """
    The duplication the extraction actually removed.

    "--url when configured" lived in three places before this: the
    graph's discovery command, the graph's per-container log command,
    and the collector's richer `podman ps`. Each needed different
    arguments after it and there was no seam to pass them through, so
    each rebuilt the base. Patching the flag in ONE place and seeing
    all three follow is what says there is one definition now -- and
    it fails the day someone rebuilds a base by hand, which is the
    only failure worth a test here.
    """

    @pytest.fixture(autouse=True)
    def _proxy_configured(self, monkeypatch):
        monkeypatch.setattr(host_exec, "SYSADMIN_PODMAN_URL", _URL)
        monkeypatch.setattr(host_exec, "SYSADMIN_JOURNAL_DIR", _DIR)

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(host_exec.discover_containers_cmd, id="discovery"),
            pytest.param(
                lambda: host_exec.collect_cmd("container", "searxng"), id="logs"
            ),
            pytest.param(lambda: containers_mod._containers_cmd(), id="collector"),
        ],
    )
    def test_every_podman_command_goes_through_the_proxy(self, build):
        assert build()[:3] == ["podman", "--url", _URL]

    @pytest.mark.parametrize("kind", ["unit", "kernel"])
    def test_every_journalctl_command_reads_the_mounted_journal(self, kind):
        assert host_exec.collect_cmd(kind, "forge.service")[:3] == [
            "journalctl",
            "-D",
            _DIR,
        ]

    def test_the_graph_sees_the_same_wiring_it_always_did(self):
        """
        The graph calls these through its own private aliases, and 48
        tests patch `sysadmin_mod._run_fixed`. That keeps working only
        while the aliases ARE the shared functions rather than copies
        that drifted back.
        """
        assert sysadmin_mod._DISCOVER_CONTAINERS_CMD() == (
            host_exec.discover_containers_cmd()
        )
        assert sysadmin_mod._collect_cmd("kernel", "") == (
            host_exec.collect_cmd("kernel", "")
        )
        assert sysadmin_mod._run_fixed is host_exec.run_fixed


def test_the_harnais_does_not_import_a_graph():
    """
    The layering the move established, asserted by importing rather
    than by reading source.

    docs/harnais.md section 1 defines the Harnais as the only layer
    that touches the real machine. A graph asking it how to reach the
    host is that sentence; the Harnais importing from a graph was its
    inverse, and it is what the collectors had to write to avoid
    duplicating the plumbing.

    This is also a real import-cost claim: forge.harnais.default_collectors
    imports its collectors inside the function specifically so that
    `import forge.harnais` stays cheap. That defence was load-bearing
    while the collectors pulled a graph in transitively. It no longer
    is, and this test is what keeps it that way.

    In a subprocess, and that is not caution for its own sake: the
    only honest way to ask "what does importing X pull in" is to start
    from nothing, and clearing forge.* out of a running session's
    sys.modules would hand every later test a module object that is no
    longer the one its fixtures patched.
    """
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "import forge.harnais.collectors.containers\n"
        "import forge.harnais.collectors.logs\n"
        "import forge.harnais.host_exec\n"
        "print(','.join(sorted(n for n in sys.modules "
        "if n.startswith('forge.graphs'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    dragged = result.stdout.strip()
    assert not dragged, f"the harnais dragged a graph in: {dragged}"
