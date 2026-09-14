"""
Route A: the Harnais behind sysadmin's no-target path.

A question that names no target ("mon deck rame", "qu'est-ce qui ne va
pas ?") used to collect `journalctl -k` and hand that block to the
model. It now runs the Harnais collectors and hands over a Context
Builder text instead -- the same kernel log, plus CPU, RAM and
container state, each marked as an observed Fact or as something that
could not be read.

Two properties carry this file, and both were settled by measurement
rather than taste (bench/context_builder_ab.py, Qwen3.8-9B,
2026-09-14):

1. A NAMED target never reaches the Harnais. Asked "pourquoi forge-llm
   plante ?" against a context saying the container could not be
   observed, this model answered "plante CAR le socket de Podman
   n'existe pas" -- the instrument's failure returned as the cause.
   Covered in tests/test_context_builder_incident.py, where the
   incident it descends from lives.
2. When NOTHING was observed, the model is not called at all. Starving
   it of evidence does not produce a refusal, it produces a worse
   answer: stripped of the failing command's error text the model
   declared HEALTHY facts guilty ("plante car la charge CPU est élevée
   (0.9 sur 1 minute)"). So the refusal is written in code.
"""

from datetime import datetime

import pytest

import forge.graphs.sysadmin as sysadmin_mod
from forge import harnais, non_answer
from forge.harnais.collectors import containers as containers_mod
from forge.harnais.collectors import cpu_ram as cpu_ram_mod
from forge.harnais.collectors import logs as logs_mod

MEMINFO = "MemTotal: 15160368 kB\nMemAvailable: 9059480 kB\n"
LOADAVG = "0.90 0.72 0.54 1/1594 2759\n"
KERNEL = "kernel: amdgpu: ring gfx timeout"

QUESTION = "Ma machine rame, qu'est-ce qui se passe ?"


def _ps(**uptimes: int) -> str:
    now = int(datetime.now().timestamp())  # noqa: DTZ005
    return "\n".join(
        f"{n}\t{now - up}\tUp {up // 60} minutes" for n, up in uptimes.items()
    )


def _fake_discover(cmd, timeout):
    """discover_node's two fixed commands, which still run on this path."""
    if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
        return '{"type":"a","data":[[["forge.service","","","","","","",0,"","/"]]]}'
    return "forge\nforge-llm"


@pytest.fixture
def healthy_machine(monkeypatch):
    """Every collector answering."""
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_discover)
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, t: _ps(forge=10800, **{"forge-llm": 247}),
    )
    monkeypatch.setattr(
        cpu_ram_mod,
        "_read",
        lambda path: MEMINFO if path == cpu_ram_mod._MEMINFO else LOADAVG,
    )
    monkeypatch.setattr(logs_mod, "_run_fixed", lambda cmd, t: KERNEL)


def _run(question=QUESTION, target=None):
    return sysadmin_mod.build().run(
        "", initial_context={"target_hint": target, "question": question}
    )


def _nodes(state) -> list[str]:
    return [step.decision_tool for step in state.trace]


# --- the path ---------------------------------------------------------------


def test_a_question_with_no_target_goes_through_the_harnais(
    healthy_machine, monkeypatch
):
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnostic")

    state = _run()

    assert _nodes(state) == ["discover", "observe", "context_synthesize"]
    assert state.final_output == "diagnostic"
    assert state.final_tool == "sysadmin"


def test_the_flag_gives_back_the_old_path_with_no_code_change(
    healthy_machine, monkeypatch
):
    """The revert path, asserted so it stays one."""
    monkeypatch.setattr(sysadmin_mod, "SYSADMIN_USE_HARNAIS", False)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnostic")

    state = _run()

    assert _nodes(state) == ["discover", "collect", "synthesize"]
    assert state.context["log_source"] == "journalctl -k"


def test_the_model_is_given_facts_it_could_not_have_had_before(
    healthy_machine, monkeypatch
):
    """
    The reason route A exists. `journalctl -k` cannot contain a
    container uptime or a memory percentage, so the old path was blind
    to both by construction -- which is why those two fixtures are
    outright losses for the log block and no amount of resampling
    changes it.
    """
    seen = {}
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: seen.setdefault("prompt", p) and "diagnostic",
    )

    _run()

    assert "ram.used_pct" in seen["prompt"]
    assert "container.forge-llm.uptime_s" in seen["prompt"]
    assert "journalctl -k" in seen["prompt"]  # the old evidence is still there


def test_the_prompt_keeps_no_think_at_position_zero(healthy_machine, monkeypatch):
    """
    Measured 2026-08-16 with bench/no_think_ab.py: removing it made this
    model return the GOOD ANSWER example instead of an answer, twice,
    deterministically. The Context Builder does not emit it, so the
    graph has to.
    """
    seen = {}
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: seen.setdefault("prompt", p) and "diagnostic",
    )

    _run()

    assert seen["prompt"].startswith("/no_think\n")


def test_today_is_stated_once_not_twice(healthy_machine, monkeypatch):
    """The Context Builder emits today_line() itself; the graph must not
    add a second one."""
    seen = {}
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: seen.setdefault("prompt", p) and "diagnostic",
    )

    _run()

    assert seen["prompt"].count("Today's date is") == 1


# --- nothing observed -------------------------------------------------------


@pytest.fixture
def blind_machine(monkeypatch):
    """Every collector failing -- proxies down, no journal, no /proc."""
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _fake_discover)
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, t: "[error] podman exited 125: no such file or directory",
    )
    monkeypatch.setattr(
        cpu_ram_mod,
        "_read",
        lambda path: (_ for _ in ()).throw(OSError("no /proc here")),
    )
    monkeypatch.setattr(
        logs_mod, "_run_fixed", lambda cmd, t: "[error] journalctl: no journal"
    )


def test_nothing_observed_does_not_call_the_model_at_all(blind_machine, monkeypatch):
    """
    The measured decision. Asked anyway, this model does not refuse --
    it reaches for whatever fact remains and declares it the cause,
    including facts that say the machine is fine.
    """

    def no_call(prompt, grammar=None):  # pragma: no cover - must not run
        raise AssertionError("the model was asked to diagnose nothing at all")

    monkeypatch.setattr(sysadmin_mod, "call_llm", no_call)

    state = _run()

    assert _nodes(state) == ["discover", "observe", "nothing_observed"]


def test_the_refusal_names_every_source_that_failed(blind_machine, monkeypatch):
    """
    Naming the command that broke is the single most useful thing a
    reader gets -- the same argument _collect_failed_node makes. The
    error text does not disappear when the model stops seeing it, it
    changes channel.
    """
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "x")

    output = _run().final_output

    assert "podman exited 125" in output
    assert "journalctl" in output
    assert "/proc" in output
    for domain in ("cpu", "ram", "container", "logs"):
        assert domain in output


def test_the_refusal_says_it_is_about_observation_not_health(
    blind_machine, monkeypatch
):
    """An empty observation is not a healthy machine, and the reply has
    to say which of the two it is reporting."""
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "x")

    assert "pas de l'état de la machine" in _run().final_output


def test_the_refusal_is_recognised_as_a_non_answer(blind_machine, monkeypatch):
    """
    non_answer.py's DRIFT rule: producers import the constants, and a
    test runs the producer rather than reading them. A refusal that
    stops matching is indexed as though it were an answer and outranks
    the real reply to its own question months later.
    """
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "x")

    assert non_answer.is_non_answer(_run().final_output)


def test_the_node_that_reports_nothing_declares_it_does_not_answer():
    """Codebase invariant: a node that ends a run without answering
    says so, rather than leaving the trace to claim a success."""
    graph = sysadmin_mod.build()

    assert graph._nodes["nothing_observed"].answers is False
    assert graph._nodes["context_synthesize"].answers is True


# --- partial failure --------------------------------------------------------


def test_one_dead_collector_does_not_stop_the_others(healthy_machine, monkeypatch):
    """
    The state this machine was really in from 09-11 to 09-14: podman
    unreachable, everything else fine. The run still answers, from what
    was observed, and the context says plainly what was not.
    """
    monkeypatch.setattr(
        containers_mod,
        "_run_fixed",
        lambda cmd, t: "[error] podman exited 125: no such file or directory",
    )
    seen = {}
    monkeypatch.setattr(
        sysadmin_mod,
        "call_llm",
        lambda p, grammar=None: seen.setdefault("prompt", p) and "diagnostic",
    )

    state = _run()

    assert _nodes(state) == ["discover", "observe", "context_synthesize"]
    assert "[unobserved] container:" in seen["prompt"]
    assert "ram.used_pct" in seen["prompt"]


def test_a_dead_collector_is_never_rendered_as_a_fact(healthy_machine, monkeypatch):
    """The whole thesis, at the last place it could still be lost."""
    from forge.kernel.context_builder import states_fact_about

    monkeypatch.setattr(
        containers_mod, "_run_fixed", lambda cmd, t: "[error] podman is gone"
    )
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnostic")

    context = _run().context["harnais_context"]

    assert not states_fact_about(context, "container")


# --- the trace the user sees ------------------------------------------------


def test_the_trace_names_the_domain_that_went_unobserved(healthy_machine, monkeypatch):
    """
    A domain that vanishes from the step detail reads as one with
    nothing to report, which is the distinction this whole path exists
    to keep. It has to be named, not omitted.
    """
    monkeypatch.setattr(
        containers_mod, "_run_fixed", lambda cmd, t: "[error] podman is gone"
    )
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnostic")

    state = _run()
    steps = sysadmin_mod._to_sub_steps(state)
    observe = next(s for s in steps if s["label"] == "observe")

    assert "non observé : container" in observe["detail"]
    assert observe["ok"] is False


def test_a_fully_observed_run_reports_a_healthy_observe_step(
    healthy_machine, monkeypatch
):
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnostic")

    steps = sysadmin_mod._to_sub_steps(_run())
    observe = next(s for s in steps if s["label"] == "observe")

    assert observe["ok"] is True
    assert "non observé" not in observe["detail"]


# --- a collector that explodes ----------------------------------------------


def test_a_collector_that_raises_does_not_take_the_run_down(
    healthy_machine, monkeypatch
):
    # Captured BEFORE the patch: calling default_collectors() inside
    # `exploding` would call `exploding` again, forever.
    real = harnais.default_collectors()

    def exploding():
        class Boom:
            name, domains = "boom", ("container",)

            def is_available(self):
                return True

            def cost_hint(self):
                return "cheap"

            def collect(self):
                raise RuntimeError("segfault in the vendor blob")

        return [Boom(), *(c for c in real if c.name != "containers")]

    monkeypatch.setattr(harnais, "default_collectors", exploding)
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnostic")

    state = _run()

    assert state.final_output == "diagnostic"
    assert "[unobserved] container:" in state.context["harnais_context"]
