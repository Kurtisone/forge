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

import subprocess

import forge.graphs.sysadmin as sysadmin_mod
from forge.graphs.sysadmin import build as build_sysadmin


def _fake_run_fixed(cmd, timeout):
    """Canned output keyed off which fixed command was requested --
    same mocking level as research's web_search.search/web_fetch.run,
    since _run_fixed is sysadmin's one external boundary."""
    if cmd == sysadmin_mod._DISCOVER_UNITS_CMD:
        return "searxng.service loaded active running\nforge.service loaded active running"
    if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD:
        return "test-container"
    if cmd[0] == "journalctl" and "-u" in cmd:
        return f"log line for unit {cmd[cmd.index('-u') + 1]}"
    if cmd[0] == "podman":
        return f"log line for container {cmd[-1]}"
    if cmd[0] == "journalctl" and "-k" in cmd:
        return "kernel log line"
    return "[unexpected cmd in test]"


def test_sysadmin_discover_lists_units_and_containers(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: "diagnosis")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert state.context["units"] == ["searxng.service", "forge.service"]
    assert state.context["containers"] == ["test-container"]


def test_sysadmin_collect_uses_unit_when_target_hint_matches_unit(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: "diagnosis")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": "searxng.service", "question": None}
    )

    assert state.context["log_source"] == "journalctl -u searxng.service"
    assert "searxng.service" in state.context["collected_logs"]


def test_sysadmin_collect_uses_container_when_target_hint_matches_container(
    monkeypatch,
):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: "diagnosis")

    state = build_sysadmin().run(
        "", initial_context={"target_hint": "test-container", "question": None}
    )

    assert state.context["log_source"] == "podman logs test-container"
    assert "test-container" in state.context["collected_logs"]


def test_sysadmin_collect_falls_back_to_kernel_when_target_not_discovered(
    monkeypatch,
):
    """Security-critical: a target_hint that doesn't appear in this
    run's own discovery output must never reach a log command -- it
    silently falls back to kernel logs instead of being trusted."""
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: "diagnosis")

    state = build_sysadmin().run(
        "",
        initial_context={"target_hint": "nginx.service", "question": None},
    )

    assert state.context["log_source"] == "journalctl -k"
    assert state.context["collected_logs"] == "kernel log line"


def test_sysadmin_collect_falls_back_to_kernel_when_no_target_hint(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: "diagnosis")

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
        if cmd[0] == "systemctl":
            return subprocess.CompletedProcess(cmd, 0, stdout="forge.service loaded active running", stderr="")
        if cmd[0] == "podman" and cmd[1] == "ps":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="kernel log", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: "diagnosis")

    hostile = "searxng; rm -rf / #"
    build_sysadmin().run("", initial_context={"target_hint": hostile, "question": None})

    for cmd in calls:
        assert hostile not in cmd
        assert all(";" not in part and "rm -rf" not in part for part in cmd)


def test_sysadmin_prompt_includes_todays_date(monkeypatch):
    from forge.context_info import today_line

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)

    captured = {}

    def fake_call_llm(prompt):
        captured["prompt"] = prompt
        return "diagnosis"

    monkeypatch.setattr(sysadmin_mod, "call_llm", fake_call_llm)
    build_sysadmin().run(
        "", initial_context={"target_hint": "searxng.service", "question": "pourquoi ça plante ?"}
    )

    assert today_line() in captured["prompt"]
    assert "pourquoi ça plante ?" in captured["prompt"]


def test_sysadmin_happy_path(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p: "Le service redémarre car le port est déjà occupé.",
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
        lambda p: (_ for _ in ()).throw(ProviderError("down")),
    )

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert not state.ok
    assert "LLM unavailable" in state.final_output


def test_sysadmin_strips_think_blocks(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod, "call_llm", lambda p: "<think>thinking...</think>Diagnosis."
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
        'le port 8888 est déjà occupé au démarrage."}'
    )
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: substantive_wrapped)

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
    huge_log = "\n".join(f"line {i} of a very long journalctl dump" for i in range(2000))
    assert len(huge_log) > sysadmin_mod.SYSADMIN_LOG_CHARS_BUDGET

    def fake_run_fixed_huge(cmd, timeout):
        if cmd == sysadmin_mod._DISCOVER_UNITS_CMD:
            return "forge.service loaded active running"
        if cmd == sysadmin_mod._DISCOVER_CONTAINERS_CMD:
            return ""
        return huge_log

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", fake_run_fixed_huge)

    captured = {}

    def fake_call_llm(prompt):
        captured["prompt"] = prompt
        return "diagnosis"

    monkeypatch.setattr(sysadmin_mod, "call_llm", fake_call_llm)

    build_sysadmin().run(
        "", initial_context={"target_hint": "forge.service", "question": None}
    )

    # the prompt must stay well under what blew the real context window
    assert len(captured["prompt"]) < 4000
    # the tail of the log (most recent lines) must be preserved
    assert "line 1999" in captured["prompt"]
    # the head must have been dropped
    assert "line 0 of a very long journalctl dump" not in captured["prompt"]


def test_sysadmin_run_publishes_sub_steps_for_the_ui(monkeypatch):
    """The top-level run() (the one tools/sysadmin.py calls) must
    publish readable sub-steps via forge.subtrace so the UI can show
    discover/collect/synthesize as expandable detail -- see
    forge/subtrace.py's docstring."""
    from forge import subtrace

    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p: "Diagnostic clair.")

    sysadmin_mod.run("searxng.service", None)

    steps = subtrace.pop()
    labels = [s["label"] for s in steps]
    assert labels == ["discover", "collect", "synthesize"]
    assert "service(s) actif(s)" in steps[0]["detail"]
    assert "journalctl -u searxng.service" in steps[1]["detail"]
    assert "caractères" in steps[2]["detail"]
    assert all(s["ok"] for s in steps)
    assert all(isinstance(s["duration_ms"], int) for s in steps)


def test_sysadmin_cleans_json_wrapped_response_like_review(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_run_fixed)
    monkeypatch.setattr(
        sysadmin_mod, "call_llm", lambda p: '{"tool":"chat","content":"query"}'
    )

    state = build_sysadmin().run(
        "", initial_context={"target_hint": None, "question": None}
    )

    assert state.final_output == '{"tool":"chat","content":"query"}'
