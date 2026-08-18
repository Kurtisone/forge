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
from typing import Protocol

from forge.errors import ForgeError
from forge.jobs import Job
from forge.logger import log


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
