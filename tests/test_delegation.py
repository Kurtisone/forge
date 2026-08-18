"""
Tests for forge.delegation: the step that runs before the router.

The property under test throughout is that none of this asks the
model anything. Whether a message is an answer to a pending question,
whether it is a cancellation, whether it approves a spec -- all three
are decided in code, because "is this an answer to what you asked?"
is exactly the judgement that has lost to a deterministic check six
times on this repo.
"""

import pytest

from forge import delegation, jobs, runner, spec
from forge.executors import EchoExecutor


@pytest.fixture(autouse=True)
def _local_runner():
    r = runner.JobRunner(EchoExecutor(), timeout=5)
    runner.set_runner(r)
    yield r
    r.stop()
    runner.set_runner(None)


def _waiting_job(field: str = "objective", spec_data: dict | None = None) -> jobs.Job:
    job = jobs.create(spec_data or {})
    return jobs.transition(job.id, jobs.AWAITING_USER, pending_field=field)


def test_nothing_pending_means_the_router_gets_the_message():
    """
    The common case, and it has to stay cheap: one read of the jobs
    file, no LLM call, and None so run() proceeds exactly as before.
    """
    assert delegation.intercept("écris-moi une fonction de tri") is None


def test_an_answer_fills_the_pending_field_and_asks_the_next_question():
    job = _waiting_job("objective")
    reply = delegation.intercept("réparer le cache KV")

    assert jobs.get(job.id).spec["objective"] == "réparer le cache KV"
    assert reply
    assert jobs.get(job.id).pending_field == "workspace"


def test_questions_are_asked_one_at_a_time_until_the_spec_is_complete():
    job = _waiting_job("objective")
    delegation.intercept("réparer le cache")
    delegation.intercept("src/forge")
    reply = delegation.intercept("les tests passent")

    assert jobs.get(job.id).pending_field == delegation.CONFIRM
    assert "Objectif" in reply
    assert "oui" in reply


def test_the_spec_is_shown_before_anything_is_queued():
    """
    The user approving the rendered spec IS the judgement step -- the
    code only ever checked that fields were non-empty. Queueing
    without showing it would remove the only place a wrong spec can be
    caught.
    """
    job = _waiting_job(delegation.CONFIRM, {"objective": "a", "workspace": "b"})
    assert jobs.get(job.id).status == jobs.AWAITING_USER

    reply = delegation.intercept("oui")
    assert "lancé" in reply


def test_approval_is_a_yes_list_not_an_interpretation():
    """
    Reading "pas encore, il manque les tests" as approval would queue
    work on a spec the user was in the middle of correcting. That is
    the one mistake this step exists to prevent.
    """
    job = _waiting_job(delegation.CONFIRM, {"objective": "a"})
    reply = delegation.intercept("pas encore, il manque les critères")

    assert jobs.get(job.id).status == jobs.AWAITING_USER
    assert jobs.get(job.id).pending_field == delegation.CONFIRM
    assert "annule" in reply


def test_a_bare_cancellation_stops_the_waiting_job():
    job = _waiting_job("workspace")
    assert "annulé" in delegation.intercept("annule")
    assert jobs.get(job.id).status == jobs.CANCELLED


@pytest.mark.parametrize(
    "word", ["annule", "Annuler", "STOP", "arrête", "laisse tomber"]
)
def test_cancellation_matching_is_accent_and_case_insensitive(word):
    job = _waiting_job("workspace")
    delegation.intercept(word)
    assert jobs.get(job.id).status == jobs.CANCELLED


def test_a_cancellation_word_inside_a_sentence_is_not_a_cancellation():
    """
    "n'annule pas les tests" contains "annule". Matching substrings
    would throw away work the user explicitly asked to keep; missing a
    real cancellation only costs them a second attempt.
    """
    job = _waiting_job("objective")
    delegation.intercept("n'annule pas les tests existants")

    assert jobs.get(job.id).status == jobs.AWAITING_USER
    assert "annule" in jobs.get(job.id).spec["objective"]


def test_cancel_outside_delegation_is_left_to_the_router():
    """
    With nothing in flight, "annule" belongs to the conversation.
    Hijacking it would make Forge unable to discuss cancelling
    anything.
    """
    assert delegation.intercept("annule") is None


def test_cancel_stops_a_job_that_is_already_queued(_local_runner):
    job = jobs.create({"objective": "a"})
    jobs.transition(job.id, jobs.READY)

    assert "annulé" in delegation.intercept("annule")
    assert jobs.get(job.id).status == jobs.CANCELLED


def test_two_live_jobs_make_the_user_choose_rather_than_guessing():
    """
    Same rule as the single-waiting-job invariant: with two
    candidates there is no deterministic answer, and guessing means
    throwing away work at random.
    """
    for _ in range(2):
        job = jobs.create({"objective": "a"})
        jobs.transition(job.id, jobs.READY)

    reply = delegation.intercept("annule")
    assert "Lequel" in reply
    assert all(j.status == jobs.READY for j in jobs.all_jobs())


def test_an_answer_that_looks_like_a_list_is_split():
    job = _waiting_job("acceptance", {"objective": "a", "workspace": "b"})
    delegation.intercept("les tests passent\nruff est propre")
    assert jobs.get(job.id).spec["acceptance"] == [
        "les tests passent",
        "ruff est propre",
    ]


def test_the_filled_spec_round_trips_through_the_spec_model():
    job = _waiting_job("objective")
    delegation.intercept("réparer le cache")
    stored = jobs.get(job.id).spec
    assert set(stored) == set(spec.FIELD_NAMES)


def test_a_short_question_is_not_recorded_as_an_answer():
    """
    From the first real run: "C'est à dire ?" was filed as the
    workspace. The interception cannot tell an answer from a question
    in general -- that is the price of deciding in code -- but this
    much is decidable, and re-asking costs a turn while recording
    garbage costs the spec.
    """
    job = _waiting_job("workspace")
    reply = delegation.intercept("C'est à dire ?")

    assert jobs.get(job.id).spec.get("workspace", "") == ""
    assert jobs.get(job.id).pending_field == "workspace"
    assert "annule" in reply


def test_a_long_answer_ending_in_a_question_mark_is_still_an_answer():
    job = _waiting_job("objective")
    delegation.intercept(
        "est-ce que tu peux regarder pourquoi la pagination casse au-delà de 50 ?"
    )
    assert jobs.get(job.id).spec["objective"]
    assert jobs.get(job.id).pending_field == "workspace"


def test_jobs_can_be_listed_from_the_thread():
    """
    GET /jobs needs a bearer token and answered 401 from a phone
    browser. Loosening the auth on an endpoint that exposes queued
    work would be the wrong fix; the listing belongs where Forge's
    interface actually is.
    """
    jobs.create({"objective": "corriger le cache KV"})
    reply = delegation.intercept("jobs")
    assert "corriger le cache KV" in reply
    assert "draft" in reply


def test_listing_jobs_when_there_are_none():
    assert "Aucun" in delegation.intercept("mes jobs")
