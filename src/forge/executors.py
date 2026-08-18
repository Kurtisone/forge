"""
Who actually carries out a delegation job.

The point of the protocol is that Forge does not care. A job is a spec
plus a lifecycle; running it is somebody else's problem, and which
somebody is the part most likely to change -- today there is no
implementer reachable from the container at all, tomorrow it could be
a CLI on the host, later a local model. Everything else in v3.13 is
built on the protocol, so that day costs one class rather than one
rewrite.

Two things the signature says on purpose:

`cancel` is an Event, not a promise to kill. A Python thread cannot
be killed from outside, so cancellation here is cooperative by
construction and an executor that never looks at the event cannot be
stopped. A subprocess-based executor gets the real mechanism -- SIGTERM
to the process group, then SIGKILL after a grace period -- and uses the
event to trigger it; it does not get to ignore it.

`deadline` is absolute (a time.time() value), not a duration. A
duration has to be re-derived after every partial wait, and each
re-derivation is a chance to hand the executor a fresh full timeout
and lose the bound entirely.
"""

import threading
import time
from pathlib import Path
from typing import Protocol

from forge.config import WORKSPACE_DIR
from forge.errors import ForgeError
from forge.jobs import Job
from forge.logger import log
from forge.spec import FIELD_NAMES, Spec, render


class JobCancelled(ForgeError):
    """The job was cancelled while running."""


class JobTimedOut(ForgeError):
    """The job passed its deadline."""


class Executor(Protocol):
    """What the runner needs from whoever does the work."""

    name: str

    def run(self, job: Job, cancel: threading.Event, deadline: float) -> str:
        """
        Carry out *job* and return what to report back.

        Raise JobCancelled once *cancel* is set, or JobTimedOut past
        *deadline*. Any other exception is a failure and is recorded
        as one.
        """
        ...


class EchoExecutor:
    """
    Does no work and says so.

    Not only a test double. It makes lots 1-4 testable end to end
    while there is no implementer reachable from the container at all,
    which is what stops that missing dependency from blocking the
    lifecycle, the interception and the cancellation behind it.

    It checks `cancel` every tick rather than sleeping once for the
    whole duration, because an executor that only notices cancellation
    when it happens to finish is indistinguishable from one that
    ignores it.
    """

    name = "echo"

    def __init__(self, duration: float = 0.0, tick: float = 0.05):
        self.duration = duration
        self.tick = tick

    def run(self, job: Job, cancel: threading.Event, deadline: float) -> str:
        ends_at = time.time() + self.duration
        while time.time() < ends_at:
            if cancel.is_set():
                raise JobCancelled(f"job {job.id} cancelled while running")
            if time.time() >= deadline:
                raise JobTimedOut(f"job {job.id} passed its deadline")
            time.sleep(min(self.tick, max(0.0, ends_at - time.time())))

        log.info("echo executor completed job %d", job.id)
        objective = job.spec.get("objective", "")
        return f"[echo] rien n'a été exécuté. Spec reçue : {objective}"


class HandoffExecutor:
    """
    Writes the spec into the workspace and stops there.

    This is the executor Forge ships with, and it is a deliberate
    choice rather than a placeholder. There is no implementer
    reachable from the container: no CLI on PATH, no Node, and the
    Claude Code CLI bundled inside the VS Code extension lives at a
    path that is not a supported entry point and moves with every
    update. Building a host-side proxy for it would be plumbing for a
    dependency already known to be temporary.

    So the last link stays manual, and everything before it does not.
    The value of delegation was never the automation of the final
    step: it was being able to describe a task from a phone, be
    interviewed about the parts that were vague, and have a checkable
    spec waiting. That works today. The spec file is what gets handed
    to an implementer at the desk.

    When a real executor exists, it implements this same protocol and
    nothing else in v3.13 changes.
    """

    name = "handoff"

    def __init__(self, directory: str | None = None):
        self.directory = directory or str(Path(WORKSPACE_DIR) / "delegations")

    def run(self, job: Job, cancel: threading.Event, deadline: float) -> str:
        if cancel.is_set():
            raise JobCancelled(f"job {job.id} cancelled before it was written")

        # int-formatted job id, fixed directory: the path is built
        # here, never taken from the spec, so nothing the model wrote
        # can steer where this lands.
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"job-{int(job.id)}.md"

        current = Spec(**{k: v for k, v in job.spec.items() if k in FIELD_NAMES})
        path.write_text(
            f"# Job {job.id}\n\n{render(current)}\n",
            encoding="utf-8",
        )
        log.info("job %d handed off at %s", job.id, path)

        return (
            f"Spec écrite dans {path}.\n"
            "Rien n'a été exécuté : donne-la à un implémenteur, "
            "puis reviens fermer le job."
        )
