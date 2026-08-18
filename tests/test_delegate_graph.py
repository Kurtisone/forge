"""
Tests for forge.graphs.delegate and the handoff executor.

The property that matters most here is that the LLM call is an
accelerator, not a dependency: every failure of it still leaves a
usable job, because the interview in delegation.py can fill the spec
on its own. Lot 7 took that to its conclusion -- the draft is off by
default -- so the tests that exercise it turn it back on explicitly.
"""

import json
import threading
import time
from pathlib import Path

import pytest

import forge.graphs.delegate as delegate_mod
from forge import jobs
from forge.errors import ProviderError
from forge.executors import HandoffExecutor, JobCancelled


@pytest.fixture
def drafting(monkeypatch):
    """Turn the optional draft call back on for the tests about it."""
    monkeypatch.setattr(delegate_mod, "DELEGATE_DRAFT", True)


def _draft(monkeypatch, **fields):
    payload = {
        "objective": "",
        "workspace": "",
        "acceptance": [],
        "constraints": [],
        "context": "",
    }
    payload.update(fields)
    monkeypatch.setattr(
        delegate_mod, "call_llm", lambda prompt, grammar=None: json.dumps(payload)
    )


def test_a_grounded_workspace_survives_the_draft(monkeypatch, drafting):
    """
    A workspace named in the request is a restatement, so the draft
    keeps it and no question is asked about it.
    """
    _draft(
        monkeypatch,
        objective="corriger le cache KV",
        workspace="src/forge",
        acceptance=["les tests passent"],
    )
    delegate_mod.run("corrige le cache KV dans src/forge")

    job = jobs.all_jobs()[0]
    assert job.spec["workspace"] == "src/forge"
    assert job.pending_field == "acceptance"


def test_acceptance_is_always_asked_however_complete_the_draft(monkeypatch, drafting):
    """
    The failure this whole lot exists for: on the first real run, two
    jobs out of three skipped the interview entirely because the model
    had invented every required field. Acceptance criteria are the
    only thing making a spec checkable, and criteria the user never
    saw are worse than none.
    """
    _draft(
        monkeypatch,
        objective="migrer les tests",
        workspace="tests",
        acceptance=["tous les tests passent en asyncio"],
        constraints=["ne pas toucher aux fixtures"],
        context="projet Python",
    )
    delegate_mod.run("migre les tests vers pytest-asyncio")

    job = jobs.all_jobs()[0]
    assert job.pending_field == "acceptance"
    assert job.spec["acceptance"] == []
    assert job.spec["constraints"] == []


def test_an_ungrounded_workspace_is_dropped(monkeypatch):
    """
    The model extrapolates more, not less, as the request gets
    specific -- an invented path reads as a complete spec and points
    an implementer at the wrong directory.
    """
    _draft(monkeypatch, objective="migrer les tests", workspace="src/legacy/api")
    delegate_mod.run("migre les tests vers pytest-asyncio")

    job = jobs.all_jobs()[0]
    assert job.spec["workspace"] == ""
    assert job.pending_field == "workspace"


def test_an_incomplete_request_starts_the_interview(monkeypatch):
    _draft(monkeypatch, objective="corriger le cache KV")
    output = delegate_mod.run("corrige le cache KV")

    job = jobs.all_jobs()[0]
    assert job.pending_field == "workspace"
    assert job.spec["objective"] == "corrige le cache KV"
    assert output


def test_the_spec_call_runs_under_the_spec_grammar(monkeypatch, drafting):
    """
    The first call in Forge constrained by something other than the
    router's grammar (lot 1). Without it the answer comes back shaped
    like a routing decision, which is why every other graph carries
    try_unwrap_router_json().
    """
    seen = {}

    def _capture(prompt, grammar=None):
        seen["grammar"] = grammar
        return json.dumps({"objective": "a"})

    monkeypatch.setattr(delegate_mod, "call_llm", _capture)
    delegate_mod.run("fais un truc")

    assert seen["grammar"] is not None
    assert "objective" in seen["grammar"]


def test_a_provider_failure_still_opens_a_job(monkeypatch, drafting):
    """
    Degraded, not failed. The interview can fill every field on its
    own, so a model that cannot draft costs questions, not the
    feature.
    """

    def _down(prompt, grammar=None):
        raise ProviderError("llama-server is down")

    monkeypatch.setattr(delegate_mod, "call_llm", _down)
    output = delegate_mod.run("corrige le cache KV")

    job = jobs.all_jobs()[0]
    # The objective survives a dead provider because it never came
    # from the model: it is the user's own message. Only workspace
    # and acceptance are left to ask.
    assert job.spec["objective"] == "corrige le cache KV"
    assert job.pending_field == "workspace"
    assert output


def test_an_unparseable_draft_still_opens_a_job(monkeypatch, drafting):
    monkeypatch.setattr(
        delegate_mod, "call_llm", lambda prompt, grammar=None: "je ne sais pas"
    )
    delegate_mod.run("corrige le cache KV")
    assert jobs.all_jobs()[0].pending_field == "workspace"


def test_a_second_job_is_refused_while_one_is_waiting(monkeypatch):
    """
    A DEFENSIVE path, not one chat can reach.

    The real traces settled this: a second delegation request sent
    while a job is waiting never gets to the router at all --
    delegation.intercept() takes it first and records it as the answer
    to the pending question. So this branch only guards direct callers
    of delegate_mod.run() (the Python entry point, this test), and it
    is kept for them rather than deleted: without it jobs.transition()
    raises JobStateError out of a tool.
    """
    _draft(monkeypatch, objective="a")
    delegate_mod.run("premier")
    output = delegate_mod.run("deuxième")

    assert len(jobs.all_jobs()) == 1
    assert "attend déjà" in output


def test_the_tool_wrapper_rejects_an_empty_request():
    from forge.tools.delegate import run as tool_run

    assert tool_run("   ").startswith("[error]")


def test_handoff_writes_the_spec_and_runs_nothing(tmp_path):
    """
    The last link stays manual because no implementer is reachable
    from the container. Everything before it does not -- the point was
    describing a task from a phone and having a checkable spec
    waiting, and that works today.
    """
    job = jobs.create({"objective": "corriger le cache KV", "workspace": "src/forge"})
    executor = HandoffExecutor(directory=str(tmp_path))

    output = executor.run(job, threading.Event(), deadline=time.time() + 60)

    written = Path(tmp_path) / f"job-{job.id}.md"
    assert written.exists()
    assert "corriger le cache KV" in written.read_text(encoding="utf-8")
    assert "Rien n'a été exécuté" in output


def test_handoff_builds_its_own_path(tmp_path):
    """
    The filename comes from the job id as an int and a fixed
    directory, never from the spec: nothing the model wrote can steer
    where this lands.
    """
    directory = tmp_path / "delegations"
    job = jobs.create({"objective": "../../etc/passwd", "workspace": "/etc"})
    HandoffExecutor(directory=str(directory)).run(
        job, threading.Event(), deadline=time.time() + 60
    )
    assert [p.name for p in directory.iterdir()] == [f"job-{job.id}.md"]


def test_handoff_honours_a_cancellation_taken_before_it_writes(tmp_path):
    directory = tmp_path / "delegations"
    cancel = threading.Event()
    cancel.set()
    job = jobs.create({"objective": "a"})

    with pytest.raises(JobCancelled):
        HandoffExecutor(directory=str(directory)).run(
            job, cancel, deadline=time.time() + 60
        )
    assert not directory.exists()


def test_the_objective_is_the_raw_message_not_the_routers_restatement():
    """
    On one real turn the user typed "délègue un truc" and the router
    answered with `content` lifted from an EARLIER message in the
    history, so the job's objective described a request nobody had
    just made. The raw message cannot drift from itself.
    """
    from forge import turn

    turn.set_input("corrige la pagination du journal")
    try:
        delegate_mod.run("quelque chose de complètement différent")
    finally:
        turn.clear()

    assert jobs.all_jobs()[0].spec["objective"] == "corrige la pagination du journal"


def test_no_llm_call_at_all_when_the_draft_is_off(monkeypatch):
    """
    The default. Across the first real delegations the draft cost
    6-14 s and only ever contributed a workspace the user had already
    typed; the interview turns it saved cost 0 ms each.
    """

    def _explode(*args, **kwargs):
        raise AssertionError("the draft must not run when DELEGATE_DRAFT is off")

    monkeypatch.setattr(delegate_mod, "call_llm", _explode)
    delegate_mod.run("corrige le cache KV dans src/forge")

    assert jobs.all_jobs()[0].pending_field == "workspace"


def test_the_draft_can_be_turned_back_on(monkeypatch, drafting):
    """
    Kept rather than deleted: what fails is the model, not the design.
    Point call_llm at something stronger and this is worth its call
    again.
    """
    _draft(monkeypatch, workspace="src/forge")
    delegate_mod.run("corrige le cache KV dans src/forge")

    job = jobs.all_jobs()[0]
    assert job.spec["workspace"] == "src/forge"
    assert job.pending_field == "acceptance"
