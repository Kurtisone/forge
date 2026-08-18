"""
Persisted delegation jobs.

A job is a spec plus a lifecycle. It exists because delegation is the
first thing in Forge that outlives the HTTP request that created it:
everything before this either finished inside run() or did not happen.
That is one property, and it drags in three consequences that all
have to be handled here rather than discovered later.

Its own file, not a key in memory.json. Compaction rewrites
memory.json in full (see compaction.py), and a job updated by the
runner while compaction is holding an older copy of that dict would
be silently dropped on the next save_memory(). Two writers, one
whole-file write, last one wins -- and the one that loses is the job.

Written through a temp file and os.replace(), unlike memory.py, which
writes in place. memory.py's docstring is right that Forge is
single-writer, and that is exactly what stops being true here: the
runner thread writes while a request thread reads. os.replace is
atomic on POSIX, so a reader sees either the old file or the new one,
never half of one.

Transitions are checked. An illegal transition quietly accepted is
how a cancelled job comes back to life, and "why did that run twice"
is a question with no answer in a log that only records the final
state.
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge.config import JOBS_FILE
from forge.errors import ForgeError
from forge.logger import log

DRAFT = "draft"
AWAITING_USER = "awaiting_user"
READY = "ready"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"

#: States a job never leaves.
TERMINAL = frozenset({DONE, FAILED, CANCELLED, INTERRUPTED})

# Cancellation is reachable from every non-terminal state on purpose:
# the user asking to stop must not depend on which half of the
# lifecycle the job happens to be in when they ask.
_ALLOWED: dict[str, frozenset[str]] = {
    DRAFT: frozenset({AWAITING_USER, READY, CANCELLED, FAILED}),
    AWAITING_USER: frozenset({AWAITING_USER, READY, CANCELLED, FAILED}),
    READY: frozenset({RUNNING, CANCELLED, FAILED}),
    RUNNING: frozenset({DONE, FAILED, CANCELLED, INTERRUPTED}),
    DONE: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
    INTERRUPTED: frozenset(),
}

_LOCK = threading.RLock()


class JobStateError(ForgeError):
    """A transition the job lifecycle does not allow."""


@dataclass
class Job:
    id: int
    status: str = DRAFT
    spec: dict = field(default_factory=dict)
    # Which spec field the pending question is about. Set together
    # with AWAITING_USER and cleared on the way out, because "waiting"
    # and "waiting for what" becoming separately true is how an answer
    # gets filed against the wrong field.
    pending_field: str | None = None
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    def to_dict(self) -> dict:
        return asdict(self)


def _path() -> Path:
    return Path(JOBS_FILE)


def _read() -> dict:
    path = _path()
    if not path.exists():
        return {"jobs": [], "next_id": 1}

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {"jobs": [], "next_id": 1}

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        # Same call as memory.py makes, and for the same reason: an
        # unreadable file must not take the runtime down. Unlike
        # memory, though, losing this one loses queued work, so it is
        # moved aside instead of being overwritten in silence.
        log.error("jobs file unreadable (%s), starting fresh: %s", path, e)
        _quarantine(path)
        return {"jobs": [], "next_id": 1}

    data.setdefault("jobs", [])
    data.setdefault("next_id", 1)
    return data


def _quarantine(path: Path) -> None:
    try:
        path.rename(path.with_suffix(f".corrupt.{int(time.time())}"))
    except OSError as e:
        log.error("could not set aside the unreadable jobs file: %s", e)


def _write(data: dict) -> None:
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        log.error("failed to write jobs file %s: %s", path, e)


def _load_jobs(data: dict) -> list[Job]:
    jobs = []
    for raw in data["jobs"]:
        try:
            jobs.append(Job(**raw))
        except TypeError as e:
            # A record from a future (or older) schema. Skipped rather
            # than fatal: one unreadable job must not hide the others.
            log.warning("skipping unreadable job record: %s", e)
    return jobs


def all_jobs() -> list[Job]:
    """Every job, oldest first."""
    with _LOCK:
        return sorted(_load_jobs(_read()), key=lambda j: j.id)


def get(job_id: int) -> Job | None:
    with _LOCK:
        for job in _load_jobs(_read()):
            if job.id == job_id:
                return job
    return None


def create(spec: dict | None = None, status: str = DRAFT) -> Job:
    with _LOCK:
        data = _read()
        job = Job(id=data["next_id"], status=status, spec=spec or {})
        data["jobs"].append(job.to_dict())
        data["next_id"] += 1
        _write(data)
    log.info("job %d created (%s)", job.id, job.status)
    return job


def save(job: Job) -> Job:
    """
    Persist a job's current contents WITHOUT changing its status.

    Splitting this from transition() is deliberate: filling in a spec
    field and moving through the lifecycle are different events, and a
    single save-everything call is how a status change sneaks in
    unchecked next to a data edit.
    """
    with _LOCK:
        data = _read()
        for i, raw in enumerate(data["jobs"]):
            if raw.get("id") == job.id:
                current = raw.get("status")
                if current != job.status:
                    raise JobStateError(
                        f"job {job.id}: save() cannot change status "
                        f"({current} -> {job.status}); use transition()"
                    )
                job.updated_at = time.time()
                data["jobs"][i] = job.to_dict()
                _write(data)
                return job
    raise JobStateError(f"job {job.id} does not exist")


def transition(job_id: int, status: str, **updates) -> Job:
    """
    Move a job to *status*, refusing anything the lifecycle forbids.

    Reading the job back from disk rather than trusting the caller's
    copy is what makes the check mean something: the runner thread and
    a request thread can both be holding a Job object from before the
    other one's write.
    """
    with _LOCK:
        data = _read()
        for i, raw in enumerate(data["jobs"]):
            if raw.get("id") != job_id:
                continue

            current = raw.get("status", DRAFT)
            if status not in _ALLOWED.get(current, frozenset()):
                raise JobStateError(
                    f"job {job_id}: {current} -> {status} is not allowed"
                )
            if status == AWAITING_USER:
                _refuse_second_waiting_job(data, job_id)

            job = Job(**raw)
            for key, value in updates.items():
                setattr(job, key, value)
            job.status = status
            if status != AWAITING_USER:
                job.pending_field = None
            job.updated_at = time.time()

            data["jobs"][i] = job.to_dict()
            _write(data)
            log.info("job %d: %s -> %s", job_id, current, status)
            return job

    raise JobStateError(f"job {job_id} does not exist")


def _refuse_second_waiting_job(data: dict, job_id: int) -> None:
    """
    At most one job may wait on the user at a time.

    Not a tidiness rule. The next user message is routed to the
    waiting job by the fact that there is exactly one of them -- with
    two, deciding which one an answer belongs to means asking the
    model to judge, which is the thing this whole design avoids.
    """
    for raw in data["jobs"]:
        if raw.get("id") != job_id and raw.get("status") == AWAITING_USER:
            raise JobStateError(
                f"job {raw['id']} is already waiting on the user; "
                f"job {job_id} cannot wait too"
            )


def awaiting_user() -> Job | None:
    """The job currently waiting on an answer, if any."""
    with _LOCK:
        for job in _load_jobs(_read()):
            if job.status == AWAITING_USER:
                return job
    return None


def reconcile() -> list[int]:
    """
    Called at startup. Marks every RUNNING job as INTERRUPTED.

    A job is RUNNING because a process was executing it, and that
    process died with the container -- there is nothing left to attach
    to, so the state on disk is a lie the moment Forge restarts.
    Nothing is resumed automatically: a half-applied job re-run from
    the top is how the same edit lands twice, and the user is right
    there in the thread to decide.
    """
    interrupted = []
    with _LOCK:
        data = _read()
        for i, raw in enumerate(data["jobs"]):
            if raw.get("status") != RUNNING:
                continue
            job = Job(**raw)
            job.status = INTERRUPTED
            job.error = "interrompu par un redémarrage de Forge"
            job.updated_at = time.time()
            data["jobs"][i] = job.to_dict()
            interrupted.append(job.id)
        if interrupted:
            _write(data)
            log.warning("jobs interrupted by a restart: %s", interrupted)
    return interrupted
