"""
The header says when sysadmin cannot see the host.

/health now carries `host_access` ({proxy: "ok" | "unobservable" |
"unknown"}); the status line turns amber and names the proxy. Two kinds
of test, for the reason test_ui_rendering.py gives: the functions are run
where deno exists, and the wiring -- that checkHealth actually reads the
field and does not paint it red -- is asserted on the source everywhere.

Red is the wrong colour on purpose. The dot goes red when /health does not
answer at all ("offline"). A lost proxy is the opposite case: Forge
answered, and what it lost is its view of the host.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import forge.api as api_mod
from tests.js.extract_renderer import health_module

INDEX = Path(api_mod.__file__).parent / "static" / "index.html"


def _source() -> str:
    return INDEX.read_text(encoding="utf-8")


def _check_health_body() -> str:
    src = _source()
    body = src[src.index("async function checkHealth") :]
    return body[: body.index("\n}\n")]


# ── Wiring: asserted on the source, runs everywhere ─────────────────────


def test_check_health_reads_host_access():
    assert "host_access" in _check_health_body()


def test_a_blind_host_paints_the_dot_amber_not_red():
    body = _check_health_body()
    try_part = body[: body.index("} catch")]
    assert "dot warn" in try_part
    assert "dot err" not in try_part, (
        "red means /health did not answer; a lost proxy is amber"
    )
    assert re.search(r"\.dot\.warn\s*\{[^}]*--amber", _source())


def test_the_warning_comes_before_the_model_name():
    """The box is 220px and ellipsises from the right, so a warning
    appended after the model name is the first thing to be cut."""
    body = _check_health_body()
    m = re.search(
        r"textContent = blind\.length \? `⚠ \$\{blind\.join\([^)]*\)\} · \$\{text\}`",
        body,
    )
    assert m, "the warning must precede ${text} in the status line"


# ── Behaviour: executed, skipped where deno is absent ───────────────────

needs_deno = pytest.mark.skipif(
    shutil.which("deno") is None, reason="no deno on this machine"
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    work = tmp_path_factory.mktemp("health")
    (work / "health.js").write_text(health_module(INDEX), encoding="utf-8")

    def _run(expression: str):
        (work / "run.js").write_text(
            'import { hostBlindness, hostBlindnessHint } from "./health.js";\n'
            f"console.log(JSON.stringify({expression}));\n",
            encoding="utf-8",
        )
        out = subprocess.run(
            ["deno", "run", "--quiet", "run.js"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout)

    return _run


@needs_deno
class TestWhichEntriesCountAsAFault:
    def test_everything_ok_is_no_fault(self, run):
        assert run("hostBlindness({podman: 'ok', systemd: 'ok'})") == []

    def test_an_unobservable_proxy_is_named(self, run):
        assert run("hostBlindness({podman: 'unobservable', systemd: 'ok'})") == [
            "podman"
        ]

    def test_both_lost_are_both_named(self, run):
        got = run("hostBlindness({podman: 'unobservable', systemd: 'unobservable'})")
        assert got == ["podman", "systemd"]

    def test_unknown_is_not_a_fault(self, run):
        """The first probe still running. Calling that a fault would light
        the header amber on every cold start."""
        assert run("hostBlindness({podman: 'unknown', systemd: 'unknown'})") == []

    @pytest.mark.parametrize("value", ["undefined", "null", "{}", "'ok'", "[]"])
    def test_a_missing_or_odd_field_is_no_fault_and_no_crash(self, run, value):
        """Sysadmin off, or no proxy configured: the key is absent, and an
        older server never had it."""
        assert run(f"hostBlindness({value})") == []


@needs_deno
class TestTheTooltip:
    def test_it_names_the_unit_to_look_at(self, run):
        hint = run("hostBlindnessHint(['podman'])")
        assert "forge-podman-ro-proxy.service" in hint
        assert "systemctl --user status" in hint

    def test_it_names_both_when_both_are_lost(self, run):
        hint = run("hostBlindnessHint(['podman', 'systemd'])")
        assert "forge-podman-ro-proxy.service" in hint
        assert "forge-dbus-proxy.service" in hint

    def test_a_proxy_it_does_not_know_falls_back_to_its_name(self, run):
        assert "gpu" in run("hostBlindnessHint(['gpu'])")
