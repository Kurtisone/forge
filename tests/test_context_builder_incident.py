"""
Run #83fc443e, replayed against the Context Builder.

WHAT HAPPENED

The podman read-only proxy was down. `podman ps` failed. The caller --
graphs.sysadmin.running_containers(), whose docstring already warns
about exactly this -- returned []. The hint "forge-llm" therefore
matched nothing, the graph fell back to kernel logs, and the model was
handed evidence about a different subsystem together with a question
about a named container. It answered fluently: a confident, entirely
invented story about a variable blocking searxng. Every step was
individually reasonable.

graphs/sysadmin.py fixed its own version of this in code -- the run
now stops at _target_missed_node instead of collecting kernel logs.
What this file pins is the other half: the Context Builder, handed
exactly the same broken world, must make no factual claim about
containers at all.

That mattered before the wiring and matters more after it. When this
file was written the graph was untouched and the Context Builder was
called by nothing, so the two fixes were independent. Both of the
graph's paths now go through the Context Builder (PR #79, #80), which
means this world -- proxy down, everything else answering -- is one a
production run actually builds, and these assertions are about what
the model is handed rather than about a module sitting on its own.

WHY THE ASSERTIONS LOOK LIKE THIS

Not `"forge-llm" not in context` -- that would pass for the wrong
reason (a context that mentions nothing is also a context that helps
nobody) and fail for the wrong reason (a kernel log line legitimately
naming the container). The claim under test is narrower and exact:
among the lines this context PRESENTS AS OBSERVED FACT, none is about
containers. fact_claims() is what reads them back.

The second assertion is the one that keeps the first honest. Silence
about containers would satisfy "no factual claim" perfectly and read,
to a model, as nothing to report. The context has to SAY the
observation failed, and say what failed.
"""

from forge import harnais
from forge.harnais.collectors import containers as containers_mod
from forge.harnais.collectors import cpu_ram as cpu_ram_mod
from forge.harnais.collectors import logs as logs_mod
from forge.harnais.collectors import units as units_mod
from forge.kernel.context_builder import (
    ContextBuilder,
    fact_claims,
    states_fact_about,
    unobserved_claims,
)
from forge.kernel.world_model import InMemoryWorldModel

#: What the question was, on the run this file replays.
QUESTION = "pourquoi forge-llm plante ?"

#: What podman says when the read-only proxy is not answering. The
#: "[error] " prefix is _run_fixed's, and it is the only thing that
#: distinguishes this from a machine with nothing running.
PROXY_DOWN = (
    "[error] podman exited 125: Error: unable to connect to Podman socket: "
    "Get http://d/v4.0.0/libpod/_ping: dial unix /run/podman/podman.sock: "
    "connect: connection refused"
)

#: Kernel logs that were collected instead, and that mention neither
#: the container nor anything to do with it -- which is precisely why
#: a diagnosis drawn from them was invention.
KERNEL_LOGS = (
    "kernel: amdgpu: RAS: optional ras ta ucode is not available\n"
    "kernel: ACPI: battery: new extension: ACPI Battery Extension\n"
    "kernel: wlan0: authenticate with 3c:37:86:1f:2a:b0"
)

#: systemd answering with nothing wrong, in the real ListUnits shape.
UNITS_ALL_WELL = (
    '{"type":"a(ssssssouso)","data":[[["forge.service","","loaded",'
    '"active","running","","/",0,"","/"]]]}'
)

MEMINFO = "MemTotal: 15160368 kB\nMemAvailable: 9059480 kB\n"
LOADAVG = "0.14 0.39 0.68 1/1594 2759\n"

BUDGET = 4000


def _broken_world(monkeypatch) -> InMemoryWorldModel:
    """The machine as it was on run #83fc443e: proxy down, everything
    else answering normally."""
    monkeypatch.setattr(
        cpu_ram_mod,
        "_read",
        lambda path: MEMINFO if path == cpu_ram_mod._MEMINFO else LOADAVG,
    )
    monkeypatch.setattr(containers_mod, "_run_fixed", lambda cmd, timeout: PROXY_DOWN)
    monkeypatch.setattr(logs_mod, "_run_fixed", lambda cmd, timeout: KERNEL_LOGS)
    # systemd itself was answering on that run; only the podman proxy
    # was down. Patched rather than left alone because an unpatched
    # collector reads the machine the suite is running on -- which is
    # how the units collector shipped with four CI failures.
    monkeypatch.setattr(units_mod, "_run_fixed", lambda cmd, timeout: UNITS_ALL_WELL)

    world = InMemoryWorldModel()
    for observation in harnais.observe(harnais.default_collectors()):
        world.record_observation(observation)
    return world


def test_a_dead_proxy_produces_no_factual_claim_about_containers(monkeypatch):
    """The assertion this whole branch is for."""
    context = ContextBuilder(_broken_world(monkeypatch)).build_for(QUESTION, BUDGET)

    for subject in ("container", "forge-llm", "podman ps", "running_count"):
        assert not states_fact_about(context, subject), (
            f"the context states a fact about {subject!r} while the proxy "
            f"that observes it was down: {fact_claims(context)}"
        )


def test_the_context_says_the_observation_failed_and_which_one(monkeypatch):
    """
    Silence would satisfy the test above and read as "nothing to
    report" -- which is the failure mode, not the fix. The absence has
    to be stated, and it has to name the command that broke: sysadmin's
    _collect_failed_node already established that naming it is the most
    useful thing a reader can be handed.
    """
    context = ContextBuilder(_broken_world(monkeypatch)).build_for(QUESTION, BUDGET)

    container_claims = [c for c in unobserved_claims(context) if "container" in c]
    assert len(container_claims) == 1
    assert "connection refused" in container_claims[0]


def test_what_did_survive_is_still_stated_as_fact(monkeypatch):
    """
    The honest answer is not an empty one. RAM, CPU and the kernel
    journal were all readable on that run and are still evidence; only
    the container domain went dark.
    """
    context = ContextBuilder(_broken_world(monkeypatch)).build_for(QUESTION, BUDGET)

    assert states_fact_about(context, "ram.used_pct")
    assert states_fact_about(context, "cpu.load_1m")
    assert states_fact_about(context, "logs.journalctl -k.lines")


def test_the_kernel_logs_are_present_but_never_as_container_evidence(monkeypatch):
    """
    The specific substitution that produced the invented diagnosis:
    kernel logs standing in for evidence about a named container. They
    are still in the context -- they are real observations -- but they
    are labelled as what they are, `journalctl -k`, and nothing in the
    fact lines connects them to a container.
    """
    context = ContextBuilder(_broken_world(monkeypatch)).build_for(QUESTION, BUDGET)

    assert "wlan0: authenticate" in context  # the logs are there
    assert all("logs." in c or "container" not in c for c in fact_claims(context))


def test_the_reader_is_told_that_unobserved_is_not_healthy(monkeypatch):
    """
    The rule that turns the marker into an answer. Measured on
    2026-09-12 with bench/sysadmin_verdict.py: asked whether an EMPTY
    log block answered "pourquoi searxng a redémarré ?", this model
    said yes. It does not infer "I could not look" from an absence; it
    has to be told, in the same context, in words.
    """
    context = ContextBuilder(_broken_world(monkeypatch)).build_for(QUESTION, BUDGET)

    assert "not evidence that the thing is healthy" in context
    assert "never a diagnosis" in context


def test_the_same_machine_with_nothing_running_reads_differently(monkeypatch):
    """
    The control arm, and the reason this is a fix rather than a
    silencing. Swap the dead proxy for a live one on an idle machine
    and the container domain becomes a FACT of zero -- the two
    situations that run #83fc443e could not tell apart now produce
    contexts that differ in exactly the right place.
    """
    broken = ContextBuilder(_broken_world(monkeypatch)).build_for(QUESTION, BUDGET)

    monkeypatch.setattr(
        containers_mod, "_run_fixed", lambda cmd, timeout: "[no output]"
    )
    idle_world = InMemoryWorldModel()
    for observation in harnais.observe(harnais.default_collectors()):
        idle_world.record_observation(observation)
    idle = ContextBuilder(idle_world).build_for(QUESTION, BUDGET)

    assert not states_fact_about(broken, "container")
    assert states_fact_about(idle, "container.running_count")
    assert any("container" in c for c in unobserved_claims(broken))
    assert not any("container:" in c for c in unobserved_claims(idle))


def test_nothing_readable_at_all_refuses_instead_of_thinning_out(monkeypatch):
    """
    The degenerate case. With every collector down there is no evidence
    of any kind, and a context that merely looked sparse would invite
    the model to fill it. It states the refusal instead.
    """
    monkeypatch.setattr(
        cpu_ram_mod, "_read", lambda path: (_ for _ in ()).throw(OSError("no /proc"))
    )
    monkeypatch.setattr(containers_mod, "_run_fixed", lambda cmd, timeout: PROXY_DOWN)
    monkeypatch.setattr(
        logs_mod, "_run_fixed", lambda cmd, timeout: "[error] journalctl: no journal"
    )
    monkeypatch.setattr(
        units_mod, "_run_fixed", lambda cmd, timeout: "[error] busctl: no bus"
    )

    answered = [
        o.collector
        for o in harnais.observe(harnais.default_collectors())
        if not o.failed
    ]
    assert not answered, f"{answered} answered: this test is reading the real machine"

    world = InMemoryWorldModel()
    for observation in harnais.observe(harnais.default_collectors()):
        world.record_observation(observation)
    context = ContextBuilder(world).build_for(QUESTION, BUDGET)

    assert fact_claims(context) == []
    assert "NOTHING WAS OBSERVED" in context
    assert len(unobserved_claims(context)) == 5


def test_the_incident_with_a_target_named_refuses_instead_of_observing(monkeypatch):
    """
    This incident's own shape, with a target named: the proxy is down,
    so discovery finds nothing, so the named target cannot be resolved.
    The run must refuse there -- not observe what little is left and
    diagnose from it.

    THIS TEST HAS BEEN WRONG TWICE, both times by outliving what it
    described, which is worth more than the assertion itself.

    It began as "sysadmin's source mentions neither harnais nor
    context_builder", scope pinned while the Context Builder was built
    in isolation. Route A wired it and retired that, on purpose.

    It then became "a named target never reaches the Harnais", which
    route C made false three days later: a named target that IS found
    now goes through the Context Builder too, measured, deliberately.
    The test kept passing the whole time -- because HERE discovery
    fails, so _target_missed_node fires first -- while its name told
    every reader that route C did not exist.

    What survives is the narrow claim, and it is the one the incident
    is about: a target that cannot be resolved refuses before any
    observation. The general form of that, over all three refusals, is
    tests/test_sysadmin_harnais_path.py's
    test_the_three_refusals_still_fire_before_any_observation.

    Asserted on the trace rather than on the source text: a source grep
    says a branch exists, never that it is wired to the right edge.
    """
    import forge.graphs.sysadmin as sysadmin_mod

    def exploding_collectors():  # pragma: no cover - must not be reached
        raise AssertionError("the Harnais ran for a question naming a target")

    monkeypatch.setattr(harnais, "default_collectors", exploding_collectors)
    monkeypatch.setattr(
        sysadmin_mod, "_run_fixed", lambda cmd, timeout: "[error] nothing here"
    )
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "diagnostic")

    state = sysadmin_mod.build().run(
        "", initial_context={"target_hint": "forge-llm", "question": QUESTION}
    )

    assert "observe" not in [step.decision_tool for step in state.trace]
    assert state.context["target_missed"] == "forge-llm"
