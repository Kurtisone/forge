"""
Tests for forge.jobs: the persisted delegation job store.

Delegation is the first thing in Forge that outlives the request that
created it, so most of what is checked here is about what happens
BETWEEN requests: an illegal transition, two writers, a restart in the
middle of a run.
"""

import json
import threading
from pathlib import Path

import pytest

from forge import jobs


def test_create_persists_and_numbers_jobs():
    first = jobs.create({"objective": "a"})
    second = jobs.create({"objective": "b"})
    assert (first.id, second.id) == (1, 2)
    assert [j.id for j in jobs.all_jobs()] == [1, 2]
    assert jobs.get(1).spec == {"objective": "a"}


def test_a_job_survives_a_fresh_read():
    """
    The whole reason this module exists rather than a dict in memory:
    the process that queued the job is not the one that will finish
    it.
    """
    jobs.create({"objective": "a"})
    assert jobs.get(1).status == jobs.DRAFT


def test_legal_transitions_are_recorded():
    job = jobs.create()
    jobs.transition(job.id, jobs.AWAITING_USER, pending_field="workspace")
    jobs.transition(job.id, jobs.READY)
    jobs.transition(job.id, jobs.RUNNING)
    assert jobs.transition(job.id, jobs.DONE, result="ok").status == jobs.DONE


def test_an_illegal_transition_is_refused():
    """
    DRAFT -> RUNNING skips the point where the user approves the spec.
    Accepting it quietly is how a job runs on a spec nobody read.
    """
    job = jobs.create()
    with pytest.raises(jobs.JobStateError):
        jobs.transition(job.id, jobs.RUNNING)


def test_a_terminal_job_cannot_be_revived():
    """
    The reason transitions are checked at all: a cancelled job coming
    back to life leaves \"why did that run twice\" with no answer in a
    log that only records final states.
    """
    job = jobs.create()
    jobs.transition(job.id, jobs.CANCELLED)
    for target in (jobs.READY, jobs.RUNNING, jobs.DONE):
        with pytest.raises(jobs.JobStateError):
            jobs.transition(job.id, target)


def test_cancellation_is_reachable_from_every_live_state():
    """
    Asking to stop must not depend on which half of the lifecycle the
    job happens to be in when the user asks.
    """
    for path in ([], [jobs.AWAITING_USER], [jobs.READY], [jobs.READY, jobs.RUNNING]):
        job = jobs.create()
        for step in path:
            jobs.transition(job.id, step)
        assert jobs.transition(job.id, jobs.CANCELLED).status == jobs.CANCELLED


def test_only_one_job_may_wait_on_the_user():
    """
    Lot 4 routes the next user message to the waiting job by the fact
    that there is exactly one. With two, deciding which one an answer
    belongs to means asking the model to judge -- the thing this
    design exists to avoid.
    """
    first = jobs.create()
    second = jobs.create()
    jobs.transition(first.id, jobs.AWAITING_USER, pending_field="objective")
    with pytest.raises(jobs.JobStateError):
        jobs.transition(second.id, jobs.AWAITING_USER, pending_field="objective")


def test_awaiting_user_finds_the_waiting_job():
    jobs.create()
    job = jobs.create()
    assert jobs.awaiting_user() is None
    jobs.transition(job.id, jobs.AWAITING_USER, pending_field="workspace")
    assert jobs.awaiting_user().id == job.id


def test_leaving_awaiting_user_clears_the_pending_field():
    """
    \"Waiting\" and \"waiting for what\" must never be separately true:
    a stale pending_field is how an answer gets filed against a field
    the user was not asked about.
    """
    job = jobs.create()
    jobs.transition(job.id, jobs.AWAITING_USER, pending_field="workspace")
    assert jobs.transition(job.id, jobs.READY).pending_field is None


def test_save_persists_data_without_touching_status():
    job = jobs.create()
    jobs.transition(job.id, jobs.AWAITING_USER, pending_field="objective")
    job = jobs.get(job.id)
    job.spec = {"objective": "réparer le cache"}
    jobs.save(job)
    assert jobs.get(job.id).spec["objective"] == "réparer le cache"
    assert jobs.get(job.id).status == jobs.AWAITING_USER


def test_save_refuses_to_smuggle_a_status_change():
    """
    Filling in a spec field and moving through the lifecycle are
    different events. A single save-everything call is exactly how an
    unchecked status change rides along next to a data edit.
    """
    job = jobs.create()
    job.status = jobs.RUNNING
    with pytest.raises(jobs.JobStateError):
        jobs.save(job)


def test_reconcile_marks_running_jobs_interrupted():
    """
    A RUNNING job on disk is a lie the moment Forge restarts: whatever
    was executing it died with the previous process.
    """
    job = jobs.create()
    jobs.transition(job.id, jobs.READY)
    jobs.transition(job.id, jobs.RUNNING)

    assert jobs.reconcile() == [job.id]
    recovered = jobs.get(job.id)
    assert recovered.status == jobs.INTERRUPTED
    assert recovered.error


def test_reconcile_does_not_resume_anything():
    """
    Nothing is restarted automatically. A half-applied job re-run from
    the top is how the same edit lands twice, and the user is right
    there in the thread to decide.
    """
    job = jobs.create()
    jobs.transition(job.id, jobs.READY)
    jobs.reconcile()
    assert jobs.get(job.id).status == jobs.READY


def test_reconcile_leaves_finished_jobs_alone():
    job = jobs.create()
    jobs.transition(job.id, jobs.READY)
    jobs.transition(job.id, jobs.RUNNING)
    jobs.transition(job.id, jobs.DONE, result="ok")
    assert jobs.reconcile() == []
    assert jobs.get(job.id).status == jobs.DONE


def test_concurrent_transitions_do_not_lose_jobs():
    """
    The runner thread writes while a request thread reads. memory.py
    can write in place because Forge is single-writer; that is exactly
    what stops being true here.
    """
    ids = [jobs.create().id for _ in range(20)]
    errors: list[Exception] = []

    def advance(job_id):
        try:
            jobs.transition(job_id, jobs.READY)
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            errors.append(e)

    threads = [threading.Thread(target=advance, args=(i,)) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(jobs.all_jobs()) == 20
    assert all(j.status == jobs.READY for j in jobs.all_jobs())


def test_an_unreadable_jobs_file_is_set_aside_not_overwritten(monkeypatch, tmp_path):
    """
    memory.py starts fresh on a corrupt file because losing memory
    costs context. Losing this file costs queued work, so the bad copy
    is kept.
    """
    path = tmp_path / "jobs.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(jobs, "JOBS_FILE", str(path))

    assert jobs.all_jobs() == []
    assert list(tmp_path.glob("jobs.corrupt.*"))


def test_a_record_from_another_schema_does_not_hide_the_others(monkeypatch, tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps({"jobs": [{"id": 1, "surprise": True}], "next_id": 2}),
        encoding="utf-8",
    )
    monkeypatch.setattr(jobs, "JOBS_FILE", str(path))
    assert jobs.all_jobs() == []

    jobs.create({"objective": "a"})
    assert [j.id for j in jobs.all_jobs()] == [2]


def test_the_write_is_atomic(monkeypatch, tmp_path):
    """
    os.replace, not a truncate-and-write: a reader must see the old
    file or the new one, never half of one.
    """
    jobs.create({"objective": "a"})
    path = Path(jobs.JOBS_FILE)
    assert json.loads(path.read_text(encoding="utf-8"))["jobs"]
    assert not list(path.parent.glob("*.tmp"))
