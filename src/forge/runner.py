"""
The thread that runs delegation jobs.

Its own thread, deliberately not api.py's ThreadPoolExecutor. That
pool has two workers and serves every chat turn; a delegation job runs
for minutes, so two of them would leave the conversation -- the only
interface Forge has -- with nothing to answer on. Work that outlives a
request does not belong in the pool that serves requests.

One worker, not a pool. There is exactly one executor, so a second
worker would only buy the ability to run two jobs against the same
implementer at once, and with it a class of interleaving bugs nothing
here needs. This is also the honest reading of the maturity ladder:
delegation asks for asynchrony, not parallelism, so it gets a job
runner and not a scheduler. Priorities, retries and arbitration stay
unbuilt until something real asks for them.
"""

import queue
import threading
import time

from forge import jobs
from forge.config import JOB_TIMEOUT
from forge.executors import EchoExecutor, Executor, JobCancelled, JobTimedOut
from forge.logger import log

_STOP = object()


class JobRunner:
    def __init__(self, executor: Executor | None = None, timeout: int | None = None):
        self.executor = executor or EchoExecutor()
        self.timeout = JOB_TIMEOUT if timeout is None else timeout
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancels: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Daemon: a job must never keep Forge from exiting. Losing one
        # mid-flight is already handled -- jobs.reconcile() marks it
        # INTERRUPTED at the next start, which is a truthful state,
        # while a runtime that will not shut down is not recoverable
        # from the thread the user is talking in.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._queue.put(_STOP)
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    # --- api ---------------------------------------------------------

    def submit(self, job_id: int) -> None:
        """Queue a READY job. The transition to RUNNING happens in the
        worker, not here: a job is not running because someone asked
        for it, it is running because a thread picked it up."""
        self.start()
        self._queue.put(job_id)

    def cancel(self, job_id: int) -> bool:
        """
        Ask a job to stop, whether it is queued or already running.

        Returns whether anything was cancelled. The transition itself
        is left to whoever owns the state at that moment: for a
        running job the worker records CANCELLED when the executor
        gives up, so that the job is never marked cancelled while its
        executor is still doing something.
        """
        job = jobs.get(job_id)
        if job is None or job.is_terminal:
            return False

        with self._lock:
            event = self._cancels.get(job_id)

        if event is not None:
            event.set()
            log.info("job %d: cancellation requested while running", job_id)
            return True

        # Queued but not started, or not queued at all. Marked
        # terminal now; the worker skips anything that is no longer
        # READY when it dequeues, which is what closes the race with a
        # job being picked up at this exact moment.
        jobs.transition(job_id, jobs.CANCELLED)
        return True

    # --- worker ------------------------------------------------------

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self._execute(int(item))
            except Exception as e:  # noqa: BLE001 - the worker must not die
                log.error("job runner: unhandled error on job %s: %s", item, e)

    def _execute(self, job_id: int) -> None:
        job = jobs.get(job_id)
        if job is None or job.status != jobs.READY:
            # Cancelled (or already handled) between submit and here.
            log.info("job %s skipped: no longer ready", job_id)
            return

        cancel = threading.Event()
        with self._lock:
            self._cancels[job_id] = cancel

        try:
            job = jobs.transition(job_id, jobs.RUNNING)
            deadline = time.time() + self.timeout
            output = self.executor.run(job, cancel, deadline)
        except JobCancelled as e:
            jobs.transition(job_id, jobs.CANCELLED, error=str(e))
        except JobTimedOut:
            jobs.transition(
                job_id,
                jobs.FAILED,
                error=f"délai dépassé ({self.timeout} s)",
            )
        except Exception as e:  # noqa: BLE001 - recorded on the job
            log.error("job %d failed: %s", job_id, e)
            jobs.transition(job_id, jobs.FAILED, error=str(e))
        else:
            # Checked after the fact too: an executor can return
            # normally on a cancellation it noticed late, and a job
            # reported as done after the user asked to stop is worse
            # than one reported as cancelled.
            if cancel.is_set():
                jobs.transition(job_id, jobs.CANCELLED, error="annulé")
            else:
                jobs.transition(job_id, jobs.DONE, result=output)
        finally:
            with self._lock:
                self._cancels.pop(job_id, None)


_runner: JobRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> JobRunner:
    """
    The process-wide runner, created on first use.

    Lazily rather than at startup so that an instance which never
    delegates never spawns the thread, and so tests can install their
    own executor before anything runs.
    """
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = JobRunner()
        return _runner


def set_runner(runner: JobRunner | None) -> None:
    """Replace the process-wide runner (tests, and lot 5's executor)."""
    global _runner
    with _runner_lock:
        _runner = runner
