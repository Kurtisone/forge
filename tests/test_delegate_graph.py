"""
Tests for forge.graphs.delegate and the handoff executor.

The property that matters most here is that the LLM call is an
accelerator, not a dependency: every failure of it still leaves a
usable job, because the interview in delegation.py can fill the spec
on its own.
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


def test_a_grounded_workspace_survives_the_draft(monkeypatch):
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


def test_acceptance_is_always_asked_however_complete_the_draft(monkeypatch):
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
    assert job.spec["objective"] == "corriger le cache KV"
    assert output


def test_the_spec_call_runs_under_the_spec_grammar(monkeypatch):
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


def test_a_provider_failure_still_opens_a_job(monkeypatch):
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
    assert job.pending_field == "objective"
    assert output


def test_an_unparseable_draft_still_opens_a_job(monkeypatch):
    monkeypatch.setattr(
        delegate_mod, "call_llm", lambda prompt, grammar=None: "je ne sais pas"
    )
    delegate_mod.run("corrige le cache KV")
    assert jobs.all_jobs()[0].pending_field == "objective"


def test_a_second_job_is_refused_while_one_is_waiting(monkeypatch):
    """
    jobs.py allows only one job to wait at a time, and it is right to.
    This turns that invariant into a sentence rather than a stack
    trace.
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
