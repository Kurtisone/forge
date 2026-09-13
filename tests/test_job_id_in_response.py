"""
The delegation job id, as a field rather than as prose.

A delegation is long-running work: the caller has to follow it after
the response comes back -- poll GET /jobs for this id and tell the user
when it finishes, app closed or not. Until this field existed, the id
was only inside the French sentence the turn answered with ("Job 12
lancé."), so the Android client recovered it with a regex over text
that breaks the day any of those five strings is reworded.

The last test in the first class is the one that matters most: it
reads the id out of the prose exactly as that client did, and asserts
the field agrees. When the wording changes, that test fails and this
one keeps passing -- which is the whole point of moving the id out of
the sentence.
"""

from __future__ import annotations

import json
import re

import pytest

from forge import current_job, delegation, jobs, runner
from forge import orchestrator as orch_mod
from forge.executors import EchoExecutor

#: The regex the Android client used before this field existed.
_PROSE = re.compile(r"Job (\d+)")


@pytest.fixture(autouse=True)
def _local_runner():
    r = runner.JobRunner(EchoExecutor(), timeout=5)
    runner.set_runner(r)
    yield r
    r.stop()
    runner.set_runner(None)


@pytest.fixture(autouse=True)
def _clean_channel():
    current_job.clear()
    yield
    current_job.clear()


def _waiting_job(field: str = "objective") -> jobs.Job:
    job = jobs.create({})
    return jobs.transition(job.id, jobs.AWAITING_USER, pending_field=field)


def _run(message: str):
    return orch_mod.Orchestrator().run(message)


class TestATurnThatIsAboutAJobSaysWhichOne:
    def test_launching_a_job_reports_its_id(self):
        """
        The turn the brief singles out: "Job N lancé." marks the
        handover to execution, which is exactly when a client needs to
        start following it.
        """
        job = _waiting_job(delegation.CONFIRM)
        result = _run("oui")

        assert "lancé" in result.output
        assert result.job_id == job.id

    def test_answering_a_question_reports_its_id(self):
        job = _waiting_job("objective")
        result = _run("réparer le cache KV")

        assert result.job_id == job.id

    def test_cancelling_reports_its_id(self):
        job = _waiting_job("objective")
        result = _run("annule")

        assert result.job_id == job.id

    def test_a_turn_that_only_re_asks_still_names_the_job(self):
        """
        Not only the turns that CHANGE the job. A client watching job
        12 gains nothing from being told this exchange was about no
        job at all.
        """
        job = _waiting_job("objective")
        result = _run("C'est à dire ?")

        assert result.job_id == job.id

    def test_the_field_agrees_with_the_prose_it_replaces(self):
        """
        The regex the Android client used, run against the same
        answer. Both readings agree today; when the sentence is
        reworded this test fails and the field keeps working, which is
        the reason the field exists.
        """
        job = _waiting_job(delegation.CONFIRM)
        result = _run("oui")

        match = _PROSE.search(result.output)
        assert match, "the prose no longer carries an id -- the field now has to"
        assert int(match.group(1)) == result.job_id == job.id


class TestATurnThatIsNotAboutAJobSaysNothing:
    def test_an_ordinary_turn_reports_no_job(self, monkeypatch):
        monkeypatch.setattr(
            orch_mod,
            "call_llm",
            lambda prompt: json.dumps({"tool": "chat", "content": "bonjour"}),
        )
        result = _run("dis-moi bonjour")

        assert result.job_id is None

    def test_listing_the_jobs_is_about_all_of_them_so_about_none(self):
        """
        `jobs` answers with every job. One id cannot say that, and
        picking one of them would be worse than saying nothing.
        """
        _waiting_job("objective")
        result = _run("jobs")

        assert result.job_id is None

    def test_the_next_turn_does_not_inherit_the_id(self, monkeypatch):
        """
        The failure clear() exists to prevent. Inherited, every answer
        after a delegation arrives tagged with a job it has nothing to
        do with -- and a client would keep polling a job that finished
        three turns ago.
        """
        _waiting_job(delegation.CONFIRM)
        first = _run("oui")
        assert first.job_id is not None

        monkeypatch.setattr(
            orch_mod,
            "call_llm",
            lambda prompt: json.dumps({"tool": "chat", "content": "bonjour"}),
        )
        assert _run("dis-moi bonjour").job_id is None

    def test_pairing_reports_no_job(self, monkeypatch):
        """The other interception, which knows nothing about jobs."""
        from forge import pairing

        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", "http://10.8.0.1:8000")
        monkeypatch.setattr(pairing, "API_TOKEN", "bearer")
        assert _run("!pair").job_id is None


class TestTheGraphThatCreatesTheJob:
    """
    The other producer, and the one where the id is least recoverable
    any other way: when the request was sent, the job did not exist
    yet, so it is not in a GET /jobs the client made beforehand.
    """

    def test_creating_a_job_reports_its_id(self):
        from forge.graphs import delegate as delegate_mod

        delegate_mod.run("délègue la correction du cache")
        created = jobs.all_jobs()[-1]

        assert current_job.pending() == created.id

    def test_refusing_because_another_job_is_waiting_names_that_job(self):
        """
        The refusal names a job the user has to deal with. A client
        that only reads prose would have to parse "Le job N attend
        déjà" -- a sixth sentence, in a different shape from the five
        others.
        """
        from forge.graphs import delegate as delegate_mod

        waiting = _waiting_job("objective")
        current_job.clear()

        output = delegate_mod.run("délègue autre chose")

        assert "attend déjà" in output
        assert current_job.pending() == waiting.id


class TestItCrossesTheHttpBoundary:
    """
    The field only earns its keep if it survives serialisation: this is
    the boundary the Android client is on the other side of.
    """

    @pytest.fixture(autouse=True)
    def _reset_rate_limiter(self):
        from forge import ratelimit

        ratelimit.reset()
        yield
        ratelimit.reset()

    def _post(self, message: str):
        from fastapi.testclient import TestClient

        import forge.api as api_mod

        return TestClient(api_mod.app).post("/chat", json={"message": message})

    def test_chat_returns_the_job_id(self):
        job = _waiting_job(delegation.CONFIRM)
        body = self._post("oui").json()

        assert body["job_id"] == job.id

    def test_no_job_serialises_as_null_not_zero(self, monkeypatch):
        """
        `usage` already holds this line in the same response. A client
        must be able to tell "no job" from "job number 0", and 0 is
        falsy in every language that will read this JSON.
        """
        monkeypatch.setattr(
            orch_mod,
            "call_llm",
            lambda prompt: json.dumps({"tool": "chat", "content": "bonjour"}),
        )
        body = self._post("dis-moi bonjour").json()

        assert "job_id" in body, "the key must be present, so its absence is readable"
        assert body["job_id"] is None

    def test_the_client_no_longer_needs_the_regex(self):
        """
        End to end, the way the app will read it: the id arrives as a
        number in a field, with no French text involved.
        """
        job = _waiting_job(delegation.CONFIRM)
        body = self._post("oui").json()

        assert isinstance(body["job_id"], int)
        assert body["job_id"] == job.id
