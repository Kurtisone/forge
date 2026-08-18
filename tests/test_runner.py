"""
Tests for forge.runner and forge.executors.

Most of these are about the two moments where a job's state and the
executor's reality can disagree: cancellation, and a deadline.
"""

import threading
import time

import pytest

from forge import jobs, runner
from forge.executors import EchoExecutor, JobCancelled, JobTimedOut


def _ready_job() -> jobs.Job:
    job = jobs.create({"objective": "a"})
    return jobs.transition(job.id, jobs.READY)


def _wait_for(job_id: int, status: str, timeout: float = 3.0) -> jobs.Job:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job.status == status:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never reached {status} (last: {jobs.get(job_id)})")


@pytest.fixture
def stopped_runner():
    made: list[runner.JobRunner] = []

    def _make(executor=None, timeout=5):
        r = runner.JobRunner(executor or EchoExecutor(), timeout=timeout)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.stop()
    runner.set_runner(None)


def test_a_submitted_job_runs_and_records_its_output(stopped_runner):
    r = stopped_runner()
    job = _ready_job()
    r.submit(job.id)
    assert "echo" in _wait_for(job.id, jobs.DONE).result


def test_the_worker_owns_the_transition_to_running(stopped_runner):
    """
    submit() does not mark the job RUNNING. A job is not running
    because someone asked for it, it is running because a thread
    picked it up -- and between those two moments it can still be
    cancelled without anything having started.
    """
    r = stopped_runner(EchoExecutor(duration=5))
    job = _ready_job()
    assert jobs.get(job.id).status == jobs.READY
    r.submit(job.id)
    _wait_for(job.id, jobs.RUNNING)


def test_cancelling_a_queued_job_stops_it_before_it_starts(stopped_runner):
    """
    The worker skips anything that is no longer READY when it
    dequeues, which is what closes the race with a job being picked up
    at the exact moment it is cancelled.
    """
    r = stopped_runner(EchoExecutor(duration=0.2))
    blocker = _ready_job()
    job = _ready_job()
    r.submit(blocker.id)
    r.submit(job.id)

    assert r.cancel(job.id)
    _wait_for(blocker.id, jobs.DONE)
    assert jobs.get(job.id).status == jobs.CANCELLED


def test_cancelling_a_running_job_stops_it(stopped_runner):
    r = stopped_runner(EchoExecutor(duration=10))
    job = _ready_job()
    r.submit(job.id)
    _wait_for(job.id, jobs.RUNNING)

    assert r.cancel(job.id)
    _wait_for(job.id, jobs.CANCELLED)


def test_a_running_job_is_not_marked_cancelled_before_its_executor_gives_up(
    stopped_runner,
):
    """
    cancel() asks; the worker records. Marking a job CANCELLED while
    its executor is still touching a workspace would put a state on
    disk that the world does not match.
    """
    started = threading.Event()
    release = threading.Event()

    class SlowToGiveUp:
        name = "slow"

        def run(self, job, cancel, deadline):
            started.set()
            cancel.wait(timeout=3)
            release.wait(timeout=3)
            raise JobCancelled("stopped")

    r = stopped_runner(SlowToGiveUp())
    job = _ready_job()
    r.submit(job.id)
    started.wait(timeout=3)

    r.cancel(job.id)
    assert jobs.get(job.id).status == jobs.RUNNING
    release.set()
    _wait_for(job.id, jobs.CANCELLED)


def test_a_late_noticed_cancellation_still_wins(stopped_runner):
    """
    An executor can return normally on a cancellation it noticed too
    late. Reporting that job as done, after the user asked to stop, is
    worse than reporting it cancelled.
    """

    class IgnoresCancellation:
        name = "stubborn"

        def run(self, job, cancel, deadline):
            cancel.wait(timeout=3)
            return "fini quand même"

    r = stopped_runner(IgnoresCancellation())
    job = _ready_job()
    r.submit(job.id)
    _wait_for(job.id, jobs.RUNNING)

    r.cancel(job.id)
    assert _wait_for(job.id, jobs.CANCELLED).result is None


def test_cancelling_a_finished_job_reports_nothing_to_cancel(stopped_runner):
    r = stopped_runner()
    job = _ready_job()
    r.submit(job.id)
    _wait_for(job.id, jobs.DONE)
    assert r.cancel(job.id) is False


def test_a_deadline_fails_the_job(stopped_runner):
    """
    The bound exists so a hung executor cannot leave a job RUNNING
    forever -- a state nothing else would ever move it out of.
    """

    class NeverFinishes:
        name = "hang"

        def run(self, job, cancel, deadline):
            while time.time() < deadline:
                time.sleep(0.01)
            raise JobTimedOut("deadline")

    r = stopped_runner(NeverFinishes(), timeout=0)
    job = _ready_job()
    r.submit(job.id)
    assert "délai" in _wait_for(job.id, jobs.FAILED).error


def test_an_executor_raising_fails_the_job_and_keeps_the_worker_alive(stopped_runner):
    class Explodes:
        name = "boom"

        def run(self, job, cancel, deadline):
            raise RuntimeError("le workspace n'existe pas")

    r = stopped_runner(Explodes())
    first = _ready_job()
    r.submit(first.id)
    assert "workspace" in _wait_for(first.id, jobs.FAILED).error

    second = _ready_job()
    r.submit(second.id)
    _wait_for(second.id, jobs.FAILED)


def test_echo_executor_checks_cancellation_while_it_waits():
    """
    An executor that only notices cancellation when it happens to
    finish is indistinguishable from one that ignores it.
    """
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(JobCancelled):
        EchoExecutor(duration=5).run(jobs.Job(id=1), cancel, deadline=time.time() + 100)


def test_echo_executor_respects_the_deadline():
    with pytest.raises(JobTimedOut):
        EchoExecutor(duration=5).run(
            jobs.Job(id=1), threading.Event(), deadline=time.time() - 1
        )


def test_the_echo_executor_duration_is_configurable(monkeypatch):
    """
    Cancelling a RUNNING job and restarting mid-run are the only two
    behaviours in v3.13 that cannot be exercised by hand: handoff
    writes a file and returns in milliseconds, so there is no window
    to cancel inside. A behaviour verified only by its own unit test
    is one nobody has actually seen work.
    """
    monkeypatch.setattr(runner, "DELEGATE_EXECUTOR", "echo")
    monkeypatch.setattr(runner, "DELEGATE_ECHO_SECONDS", 42.0)
    executor = runner._configured_executor()
    assert executor.name == "echo"
    assert executor.duration == 42.0
