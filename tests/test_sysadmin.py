"""Tests for forge.graphs.sysadmin (discover -> collect -> synthesize).

Same posture as test_research.py: one deterministic call runs the
whole sequence, so tests focus on each node in isolation plus the
fixed edges. The one thing that matters more here than in research is
security: collect_node must NEVER pass an unvalidated target_hint to
a subprocess. That gets its own explicit test
(test_sysadmin_never_passes_unvalidated_name_to_subprocess) which
patches subprocess.run itself, not the _run_fixed boundary, so a
regression that reintroduces an injection path can't hide behind a
mocked-away _run_fixed.
"""

import json
import subprocess

import forge.graphs.sysadmin as sysadmin_mod
from forge.graphs.sysadmin import build as build_sysadmin


def _fake_busctl_units_json(names: list[str]) -> str:
    """Builds a real busctl --json=short ListUnits shape (verified
    against actual production output before writing this, not
    guessed): {"type": "a(ssssssouso)", "data": [[[10-field tuple], ...]]}."""
    rows = [
        [
            n,
            n,
            "loaded",
            "active",
            "running",
            "",
            f"/org/freedesktop/systemd1/unit/{n}",
            0,
            "",
            "/",
        ]
        for n in names
    ]
    return json.dumps({"type": "a(ssssssouso)", "data": [rows]})


def _fake_run_fixed(cmd, timeout):
    """Canned output keyed off which fixed command was requested --
    same mocking level as research's web_search.search/web_fetch.run,
    since _run_fixed is sysadmin's one external boundary."""
    if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
        return _fake_busctl_units_json(["searxng.service", "forge.service"])
    if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
        return "test-container"
    unit_flags = [arg for arg in cmd if arg.startswith("--unit=")]
    if cmd[0] == "journalctl" and unit_flags:
        return f"log line for unit {unit_flags[0].removeprefix('--unit=')}"
    if cmd[0] == "podman":
        return f"log line for container {cmd[-1]}"
    if cmd[0] == "journalctl" and "-k" in cmd:
        return "kernel log line"
    return "[unexpected cmd in test]"


def test_sysadmin_discover_lists_units_and_containers(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnosis")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert state.context["units"] == ["searxng.service", "forge.service"]
    assert state.context["containers"] == ["test-container"]


def test_sysadmin_collect_uses_unit_when_target_hint_matches_unit(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnosis")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": "searxng.service", "question": None}
    )

    assert state.context["log_source"] == "journalctl -u searxng.service"
    assert "searxng.service" in state.context["collected_logs"]


def test_sysadmin_collect_uses_container_when_target_hint_matches_container(
    monkeypatch,
):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnosis")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": "test-container", "question": None}
    )

    assert state.context["log_source"] == "podman logs test-container"
    assert "test-container" in state.context["collected_logs"]


def test_sysadmin_stops_when_the_target_is_not_discovered(monkeypatch):
    """
    Security-critical, and unchanged: a target_hint that doesn't appear
    in this run's own discovery output must never reach a log command.

    What changed is what happens instead. It used to fall back to
    kernel logs and say nothing about it, and on run #83fc443e that
    produced a confident, entirely invented diagnosis of a service the
    collected logs never mentioned. Kernel logs cannot answer a
    question about a named service, so the run now stops rather than
    handing a model evidence about the wrong subject.
    """
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)

    def no_call(prompt, grammar=None):  # pragma: no cover - must not run
        raise AssertionError("the model was asked to diagnose the wrong logs")

    monkeypatch.setattr(sysadmin_mod, "call_llm", no_call)

    state = build_sysadmin().run(
        "",
        initial_context={"target_hint": "nginx.service", "question": None},
    )

    assert "collected_logs" not in state.context
    assert state.context["target_missed"] == "nginx.service"
    assert "nginx.service" in state.final_output
    assert "introuvable" in state.final_output


def test_sysadmin_suggests_close_names_for_a_missed_target(monkeypatch):
    """A typo is the cheapest cause of a miss and the cheapest to fix."""
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnosis")

    state = build_sysadmin().run(
        "",
        initial_context={"target_hint": "searxng.servce", "question": None},
    )

    assert "searxng.service" in state.final_output


def test_a_missed_target_reports_failed_discovery_as_the_likelier_cause(monkeypatch):
    """
    The real chain on run #83fc443e was: podman proxy down -> zero
    containers -> the hint matched nothing. "I could not find your
    container" is true and useless; "container discovery failed" is
    what the user has to act on.
    """

    def broken_containers(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
            return _fake_busctl_units_json(["forge.service"])
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
            return "[error] connection refused"
        return "kernel log line"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", broken_containers)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnosis")

    state = build_sysadmin().run(
        "",
        initial_context={"target_hint": "forge-llm", "question": "pourquoi ?"},
    )

    assert "connection refused" in state.final_output


def test_sysadmin_collect_falls_back_to_kernel_when_no_target_hint(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnosis")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert state.context["log_source"] == "journalctl -k"


def test_sysadmin_never_passes_unvalidated_name_to_subprocess(monkeypatch):
    """Patches subprocess.run itself (not _run_fixed) so a regression
    that reintroduces an injection path in collect_node can't hide
    behind a mocked-away boundary. A hostile target_hint containing
    shell metacharacters must never appear in any argument list handed
    to subprocess.run, and shell=True must never be used."""
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs.get("shell") is not True
        if cmd[0] == "busctl":
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_fake_busctl_units_json(["forge.service"]), stderr=""
            )
        if cmd[0] == "podman" and cmd[1] == "ps":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="kernel log", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnosis")

    hostile = "searxng; rm -rf / #"
    build_sysadmin().run("", initial_context={"target_hint": hostile, "question": None})

    for cmd in calls:
        assert hostile not in cmd
        assert all(";" not in part and "rm -rf" not in part for part in cmd)


def test_sysadmin_prompt_includes_todays_date(monkeypatch):
    from forge.context_info import today_line

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)

    captured = {}

    def fake_call_llm(prompt, grammar=None):
        captured["prompt"] = prompt
        return "diagnosis"

    monkeypatch.setattr(sysadmin_mod, "call_llm", fake_call_llm)
    build_sysadmin().run(
        "",
        initial_context={
            "target_hint": "searxng.service",
            "question": "pourquoi ça plante ?",
        },
    )

    assert today_line() in captured["prompt"]
    assert "pourquoi ça plante ?" in captured["prompt"]


def test_sysadmin_happy_path(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: "Le service redémarre car le port est déjà occupé.",
    )

    state = build_sysadmin().run(
        "", initial_context={"target_hint": "searxng.service", "question": None}
    )

    assert state.ok
    assert state.final_output == "Le service redémarre car le port est déjà occupé."
    assert state.final_tool == "sysadmin"


def test_sysadmin_llm_unavailable(monkeypatch):
    from forge.errors import ProviderError

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: (_ for _ in ()).throw(ProviderError("down")),
    )

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert not state.ok
    assert "LLM unavailable" in state.final_output


def test_sysadmin_strips_think_blocks(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: "<think>thinking...</think>Diagnosis.",
    )

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert state.final_output == "Diagnosis."


def test_sysadmin_unwraps_substantive_json_wrapped_answer(monkeypatch):
    """Same regression class already hit once on research's first real
    run: a genuine multi-sentence diagnosis wrapped in
    {"tool":"chat","content":"..."} must be unwrapped to clean prose."""
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)

    substantive_wrapped = (
        '{"tool":"chat","content":"Le service redémarre en boucle car '
        'le disque /var est plein à 100%."}'
    )
    monkeypatch.setattr(
        sysadmin_mod, "call_llm", lambda p, grammar=None: substantive_wrapped
    )

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert state.final_output.startswith("Le service redémarre en boucle")
    assert '"tool"' not in state.final_output


def test_sysadmin_truncates_oversized_log_block(monkeypatch):
    """Regression test for the real production crash: llama.cpp
    rejected a request at 4362 tokens against a 4096-token context
    because SYSADMIN_MAX_LOG_LINES=200 alone didn't bound prompt size.
    The log block actually inserted into the prompt must respect
    SYSADMIN_LOG_CHARS_BUDGET regardless of how many lines were
    collected, and must keep the END of the log (most recent, most
    relevant events), not the start."""
    huge_log = "\n".join(
        f"line {i} of a very long journalctl dump" for i in range(2000)
    )
    assert len(huge_log) > sysadmin_mod.SYSADMIN_LOG_CHARS_BUDGET

    def fake_run_fixed_huge(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
            return _fake_busctl_units_json(["forge.service"])
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
            return ""
        return huge_log

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed_huge)

    captured = {}

    def fake_call_llm(prompt, grammar=None):
        captured["prompt"] = prompt
        return "diagnosis"

    monkeypatch.setattr(sysadmin_mod, "call_llm", fake_call_llm)

    build_sysadmin().run(
        "", initial_context={"target_hint": "forge.service", "question": None}
    )

    # The prompt is a fixed template plus a log block capped at
    # SYSADMIN_LOG_CHARS_BUDGET. Asserting that relation rather than a
    # round number: the old `< 4000` was a proxy for it and had to be
    # revisited the first time the template gained a paragraph, which
    # tells you nothing about whether the cap still holds.
    empty_template = sysadmin_mod._SYNTHESIS_PROMPT.format(
        today_line="Today's date is 2026-08-21.",
        question="Diagnostique le problème et propose une solution.",
        source="journalctl -u forge.service",
        log_block="",
    )
    assert len(captured["prompt"]) <= (
        len(empty_template) + sysadmin_mod.SYSADMIN_LOG_CHARS_BUDGET + 200
    )
    # the tail of the log (most recent lines) must be preserved
    assert "line 1999" in captured["prompt"]
    # the head must have been dropped
    assert "line 0 of a very long journalctl dump" not in captured["prompt"]


def test_run_fixed_prefixes_error_on_nonzero_exit(monkeypatch):
    """Regression test for the exact bug hit in production on
    2026-08-11: _run_fixed captured stdout/stderr on a FAILED command
    (real exit code != 0) exactly like a successful one -- only actual
    Python exceptions (FileNotFoundError/TimeoutExpired) got the
    "[error]" prefix. systemctl's real two-line failure message
    ("System has not been booted with systemd...\nFailed to connect
    to bus...") slipped through as valid output and got parsed as two
    fake unit names ("System", "Failed") by _discover_node -- this is
    also why systemctl was replaced by busctl for discovery (see
    _DISCOVER_UNITS_CMD's docstring), but the underlying _run_fixed
    exit-code bug applies to any command, not just that one."""

    class FakeCompletedProcess:
        returncode = 1
        stdout = (
            "System has not been booted with systemd as init system "
            "(PID 1). Can't operate.\nFailed to connect to bus: Host is down"
        )
        stderr = ""

    monkeypatch.setattr(
        sysadmin_mod.subprocess, "run", lambda *a, **kw: FakeCompletedProcess()
    )

    result = sysadmin_mod._run_fixed(["busctl", "call"], 10)

    assert result.startswith("[error]")
    assert "System has not been booted" in result


def test_sysadmin_discover_handles_nonzero_exit_gracefully(monkeypatch):
    """End-to-end version of the above: a real command failure (exit
    code != 0, no Python exception) must not be parsed as fake unit/
    container names, exactly like the FileNotFoundError case already
    covered by test_sysadmin_discover_handles_missing_executables_gracefully."""

    def fake_run_fixed_nonzero_exit(cmd, timeout):
        if cmd[0] == "busctl":
            return (
                "[error] busctl exited 1: System has not been booted "
                "with systemd as init system (PID 1). Can't operate.\n"
                "Failed to connect to bus: Host is down"
            )
        if cmd[0] == "podman":
            return (
                "[error] podman exited 125: Cannot connect to Podman. "
                "Error: unable to connect to Podman socket: dial unix "
                "/run/forge-podman-ro-proxy/sock: connect: no such file "
                "or directory"
            )
        return "kernel log line"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed_nonzero_exit)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "Diagnostic.")

    from forge import subtrace

    sysadmin_mod.run(None, None)
    steps = subtrace.pop()

    discover_detail = steps[0]["detail"]
    assert "System" not in discover_detail.split("erreur (")[0]
    assert "Failed" not in discover_detail.split("erreur (")[0]
    assert "has not been booted" in discover_detail
    assert "no such file or directory" in discover_detail
    assert steps[0]["ok"] is False

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )
    assert state.context["units"] == []
    assert state.context["containers"] == []


def test_sysadmin_collect_flags_error_in_sub_steps(monkeypatch):
    """The collect step must be flagged ok=False when the underlying
    command failed (e.g. podman couldn't reach its socket) -- before
    this fix only the discover step could be flagged, even though the
    exact same production case showed a failing collect too."""

    def fake_run_fixed(cmd, timeout):
        if cmd[0] == "busctl":
            return _fake_busctl_units_json(["forge.service"])
        if cmd[0] == "podman" and "ps" in cmd:
            return "test-container"
        if cmd[0] == "podman" and "logs" in cmd:
            return "[error] podman exited 125: connection refused"
        return "kernel log"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "Diagnostic.")

    from forge import subtrace

    sysadmin_mod.run("test-container", None)
    steps = subtrace.pop()

    collect_step = next(s for s in steps if s["label"] == "collect")
    assert collect_step["ok"] is False
    assert "connection refused" in collect_step["detail"]


def test_sysadmin_discover_handles_missing_executables_gracefully(monkeypatch):
    """Regression test for the exact bug hit in production: journalctl/
    podman/systemctl aren't necessarily installed inside Forge's own
    container image. _run_fixed's error string ("[error] executable
    not found: 'podman'") must never be parsed as if it were a real
    unit/container name -- it must produce an empty discovery list and
    a visible error, not a fake entry literally named "[error]"."""

    def fake_run_fixed_missing_binaries(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
            return "[error] executable not found: 'busctl'"
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
            return "[error] executable not found: 'podman'"
        return "kernel log line"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed_missing_binaries)
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: "Diagnostic sur les logs kernel.",
    )

    from forge import subtrace

    sysadmin_mod.run(None, None)
    steps = subtrace.pop()

    discover_step = steps[0]
    assert (
        "[error]" not in discover_step["detail"].split("erreur (")[0]
    )  # no fake entry before the error label
    assert "executable not found: 'busctl'" in discover_step["detail"]
    assert "executable not found: 'podman'" in discover_step["detail"]
    assert (
        discover_step["ok"] is False
    )  # flagged even though the overall run still succeeds

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )
    # kernel-log fallback keeps the run useful even when discovery failed entirely
    assert state.context["units"] == []
    assert state.context["containers"] == []


def test_sysadmin_uses_configured_journal_dir(monkeypatch):
    """SYSADMIN_JOURNAL_DIR (deploy/README.md: host's /var/log/journal
    bind-mounted read-only) must add -D <dir> to every journalctl
    call, not just kernel logs."""
    monkeypatch.setattr(sysadmin_mod, "SYSADMIN_JOURNAL_DIR", "/host-journal")
    calls = []

    def fake_run_fixed(cmd, timeout):
        calls.append(cmd)
        if cmd[0] == "busctl":
            return _fake_busctl_units_json(["forge.service"])
        if cmd[0] == "podman":
            return ""
        return "kernel log"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "Diagnostic.")

    sysadmin_mod.run("forge.service", None)

    journalctl_calls = [c for c in calls if c[0] == "journalctl"]
    assert journalctl_calls, "expected at least one journalctl call"
    for c in journalctl_calls:
        assert c[1:3] == ["-D", "/host-journal"]


def test_sysadmin_uses_configured_podman_url(monkeypatch):
    """SYSADMIN_PODMAN_URL (deploy/README.md: podman_ro_proxy.py's
    socket, never the raw host socket) must add --url <value> to
    every podman call."""
    monkeypatch.setattr(
        sysadmin_mod, "SYSADMIN_PODMAN_URL", "unix:///run/forge-podman-ro-proxy/sock"
    )
    calls = []

    def fake_run_fixed(cmd, timeout):
        calls.append(cmd)
        if cmd[0] == "busctl":
            return _fake_busctl_units_json([])
        if cmd[0] == "podman" and "ps" in cmd:
            return "test-container"
        return "container log"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "Diagnostic.")

    sysadmin_mod.run("test-container", None)

    podman_calls = [c for c in calls if c[0] == "podman"]
    assert podman_calls, "expected at least one podman call"
    for c in podman_calls:
        assert c[1:3] == ["--url", "unix:///run/forge-podman-ro-proxy/sock"]


def test_sysadmin_passes_configured_dbus_address_to_subprocess_env(monkeypatch):
    """SYSADMIN_DBUS_ADDRESS (deploy/README.md: forge-dbus-proxy.sh's
    filtered bus, never the real host system bus) must reach systemctl
    via DBUS_SYSTEM_BUS_ADDRESS in the subprocess env, and the minimal
    env posture (no host env leaking through) must be preserved."""
    monkeypatch.setattr(
        sysadmin_mod, "SYSADMIN_DBUS_ADDRESS", "unix:path=/run/forge-dbus-proxy/bus"
    )

    env = sysadmin_mod._subprocess_env()
    assert env["DBUS_SYSTEM_BUS_ADDRESS"] == "unix:path=/run/forge-dbus-proxy/bus"
    assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert "HOME" not in env or env.get("SECRET") is None  # no unexpected leakage


def test_sysadmin_no_proxy_configured_leaves_commands_unchanged(monkeypatch):
    """Default (empty) config -- the case every other test in this
    file already exercises -- must produce byte-identical commands to
    before this became configurable at all."""
    assert sysadmin_mod._DISCOVER_UNITS_CMD() == [
        "busctl",
        "--json=short",
        "call",
        "org.freedesktop.systemd1",
        "/org/freedesktop/systemd1",
        "org.freedesktop.systemd1.Manager",
        "ListUnits",
    ]
    assert sysadmin_mod._DISCOVER_CONTAINERS_CMD() == [
        "podman",
        "ps",
        "--format",
        "{{.Names}}",
    ]
    assert sysadmin_mod._collect_cmd("kernel", "") == [
        "journalctl",
        "-k",
        "--no-pager",
        "-n",
        str(sysadmin_mod.SYSADMIN_MAX_LOG_LINES),
    ]
    assert "DBUS_SYSTEM_BUS_ADDRESS" not in sysadmin_mod._subprocess_env()


def test_sysadmin_discover_units_cmd_includes_address_when_configured(monkeypatch):
    """Unlike systemctl (which ignores DBUS_SYSTEM_BUS_ADDRESS
    entirely -- see _DISCOVER_UNITS_CMD's docstring), busctl takes the
    proxy address as an explicit CLI flag, confirmed against real
    production output to actually work."""
    monkeypatch.setattr(
        sysadmin_mod, "SYSADMIN_DBUS_ADDRESS", "unix:path=/run/forge-dbus-proxy/bus"
    )
    cmd = sysadmin_mod._DISCOVER_UNITS_CMD()
    assert "--address=unix:path=/run/forge-dbus-proxy/bus" in cmd


def test_parse_busctl_units_extracts_names_from_real_shape():
    """Parses the exact busctl --json=short shape confirmed against
    real production output: {"type": "a(ssssssouso)", "data": [[10-field tuples]]},
    unit name at index 0 of each tuple."""
    raw = _fake_busctl_units_json(
        ["cups.service", "forge.service", "searxng-something.service"]
    )
    assert sysadmin_mod._parse_busctl_units(raw) == [
        "cups.service",
        "forge.service",
        "searxng-something.service",
    ]


def test_parse_busctl_units_handles_empty_list():
    raw = _fake_busctl_units_json([])
    assert sysadmin_mod._parse_busctl_units(raw) == []


def test_parse_busctl_units_raises_on_malformed_json():
    import pytest

    with pytest.raises(json.JSONDecodeError):
        sysadmin_mod._parse_busctl_units("not json at all")


def test_parse_busctl_units_raises_on_unexpected_shape():
    import pytest

    with pytest.raises((KeyError, IndexError, TypeError)):
        sysadmin_mod._parse_busctl_units(json.dumps({"unexpected": "shape"}))


def test_sysadmin_run_publishes_sub_steps_for_the_ui(monkeypatch):
    """The top-level run() (the one tools/sysadmin.py calls) must
    publish readable sub-steps via forge.subtrace so the UI can show
    discover/collect/synthesize as expandable detail -- see
    forge/subtrace.py's docstring."""
    from forge import subtrace

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod, "call_llm", lambda p, grammar=None: "Diagnostic clair."
    )

    sysadmin_mod.run("searxng.service", None)

    steps = subtrace.pop()
    labels = [s["label"] for s in steps]
    assert labels == ["discover", "collect", "synthesize"]
    assert (
        "searxng.service" in steps[0]["detail"]
        and "forge.service" in steps[0]["detail"]
    )
    assert "test-container" in steps[0]["detail"]
    assert "journalctl -u searxng.service" in steps[1]["detail"]
    assert "caractères" in steps[2]["detail"]
    assert all(s["ok"] for s in steps)
    assert all(isinstance(s["duration_ms"], int) for s in steps)


def test_sysadmin_discover_detail_caps_long_lists(monkeypatch):
    """A host with many active units shouldn't dump a wall of text
    into the UI's step detail -- _format_discovered_list caps the
    shown names and summarizes the rest."""
    many_units_names = [f"unit{i}.service" for i in range(20)]

    def fake_run_fixed_many(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
            return _fake_busctl_units_json(many_units_names)
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
            return ""
        return "kernel log"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed_many)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "Diagnostic.")

    from forge import subtrace

    sysadmin_mod.run(None, None)
    steps = subtrace.pop()

    discover_detail = steps[0]["detail"]
    assert "unit0.service" in discover_detail
    assert "unit19.service" not in discover_detail  # past the cap
    assert "+12" in discover_detail  # 20 units, 8 shown, 12 remaining


def test_sysadmin_rejects_verbatim_example_leak(monkeypatch):
    """Regression test for the exact bug hit in production on
    2026-08-11: the model copied the GOOD ANSWER example's content
    verbatim (searxng/port 8888) as its "diagnosis" for a completely
    unrelated question about a different container -- fabricated
    misinformation presented as a real answer, and _PROMPT_LEAK_MARKERS
    alone didn't catch it because the model never echoed the literal
    marker word "GOOD ANSWER:", only the content after it."""
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: (
            "Le service searxng redémarre en boucle car le port 8888 "
            "est déjà occupé au démarrage d'après les lignes \"address "
            'already in use". Je te propose de vérifier quel processus '
            "occupe ce port avant de relancer le service."
        ),
    )

    state = build_sysadmin().run(
        "",
        initial_context={
            "target_hint": "forge.service",
            "question": "logs de forge ?",
        },
    )

    assert "[error]" in state.final_output
    assert "searxng" not in state.final_output
    assert "8888" not in state.final_output


def test_sysadmin_cleans_json_wrapped_response_like_review(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: '{"tool":"chat","content":"query"}',
    )

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert state.final_output == '{"tool":"chat","content":"query"}'


# ─── Argument shape (audit M-4) ─────────────────────────────────────


def test_collect_cmd_attaches_the_unit_name_to_its_flag():
    """`-u foo` puts the name in a position where a leading dash is
    read as an option: `-u --output=cat` would be journalctl's flag,
    not a unit name. The attached `--unit=<name>` form has no such
    position -- everything after the `=` is the value, dash or not."""
    cmd = sysadmin_mod._collect_cmd("unit", "--output=cat")
    assert "--unit=--output=cat" in cmd
    assert "-u" not in cmd
    # And nothing that looks like a loose option was introduced.
    assert "--output=cat" not in cmd


def test_collect_cmd_puts_a_container_name_after_the_options_marker():
    """podman takes the container as a positional argument, so a name
    starting with a dash is parsed as a flag. `--` ends option parsing:
    whatever follows is a name, even if it's spelled `--help`."""
    cmd = sysadmin_mod._collect_cmd("container", "--help")
    assert cmd[-2:] == ["--", "--help"]


def test_ordinary_names_are_unaffected():
    """The hardening must not change what the normal path actually
    runs -- these two commands are what production executes."""
    unit_cmd = sysadmin_mod._collect_cmd("unit", "searxng.service")
    assert "--unit=searxng.service" in unit_cmd
    assert unit_cmd[0] == "journalctl"

    container_cmd = sysadmin_mod._collect_cmd("container", "forge")
    assert container_cmd[0] == "podman"
    assert container_cmd[-1] == "forge"


def test_kernel_collection_takes_no_name_at_all():
    """The kernel branch never interpolates anything, which is why it
    is the safe fallback for an unrecognised target_hint."""
    cmd = sysadmin_mod._collect_cmd("kernel", "ignored")
    assert "ignored" not in cmd
    assert "-k" in cmd


def test_the_synthesis_prompt_allows_the_logs_to_be_off_topic(monkeypatch):
    """
    Defect (4) from the v3.11 list, reproduced live on run #83fc443e:
    given the right logs or the wrong ones, this prompt asked for a
    diagnosis and never for an admission. `research` has always had
    that permission; this one did not.

    Wording, and wording loses here -- the deterministic half of the
    same defect is the target_missed node above, which removes the
    commonest way of arriving with the wrong logs in the first place.
    """
    captured = {}

    def fake_call_llm(prompt, grammar=None):
        captured["prompt"] = prompt
        return "diagnostic"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", fake_call_llm)

    build_sysadmin().run(
        "",
        initial_context={"target_hint": "forge.service", "question": "pourquoi ?"},
    )

    prompt = captured["prompt"]
    assert "may simply not contain the answer" in prompt
    assert "say so plainly" in prompt
    assert "absence of an error" in prompt


def test_a_failed_collection_never_reaches_the_model(monkeypatch):
    """
    Runs #7a29f59d and #7e0ed90c, both live on the Deck. `podman logs`
    came back as the read-only proxy's 403 refusal rather than logs,
    _run_fixed prefixed it "[error]" exactly as designed -- and the
    text went into the prompt under a "collected logs" header anyway.

    The model then explained, at length and convincingly, that the
    container was crashing BECAUSE of that 403. The container was
    healthy (Up 6 minutes). The fault did not exist.

    Same shape as target_missed, so the same answer: never hand the
    model evidence about a different subject than the question.
    """

    def failing_collect(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
            return _fake_busctl_units_json(["forge.service"])
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
            return "searxng"
        return "[error] Error: unmarshalling error into &errorhandling.ErrorModel"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", failing_collect)

    def no_call(prompt, grammar=None):  # pragma: no cover - must not run
        raise AssertionError("the model was asked to diagnose a collection error")

    monkeypatch.setattr(sysadmin_mod, "call_llm", no_call)

    state = build_sysadmin().run(
        "",
        initial_context={"target_hint": "searxng", "question": "pourquoi ?"},
    )

    assert "[collecte impossible]" in state.final_output
    assert "searxng" in state.final_output
    assert "unmarshalling" in state.final_output


def test_the_failed_collection_report_names_the_command(monkeypatch):
    """
    The failure that produced this node was a proxy policy refusal
    rendered by podman as an unmarshalling error. The single most
    useful thing to be told was which command to run by hand.
    """

    def failing_collect(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
            return _fake_busctl_units_json(["forge.service"])
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
            return ""
        return "[error] command timed out after 20s"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", failing_collect)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "x")

    state = build_sysadmin().run(
        "",
        initial_context={"target_hint": "forge.service", "question": "pourquoi ?"},
    )

    assert "journalctl -u forge.service" in state.final_output


def test_the_failed_collection_report_does_not_claim_the_target_is_broken(monkeypatch):
    """
    The exact confusion in the two observed runs: a message about the
    COLLECTION being read as a message about the TARGET.
    """

    def failing_collect(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
            return _fake_busctl_units_json(["forge.service"])
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD():
            return "searxng"
        return "[error] forbidden"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", failing_collect)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "x")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": "searxng", "question": "pourquoi ?"}
    )

    assert "pas de l'état" in state.final_output


def test_a_healthy_collection_still_reaches_the_model(monkeypatch):
    """The control. Nothing above may cost a working run."""
    captured = {}

    def fake_call_llm(prompt, grammar=None):
        captured["prompt"] = prompt
        return "diagnostic normal"

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", fake_call_llm)

    state = build_sysadmin().run(
        "", initial_context={"target_hint": "forge.service", "question": "pourquoi ?"}
    )

    assert "prompt" in captured
    assert state.final_output == "diagnostic normal"
