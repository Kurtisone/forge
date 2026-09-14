"""
Which containers are running, via the read-only podman proxy.

THE ONE THING THIS MODULE MUST NOT DO

`graphs.sysadmin.running_containers()` already answers this question,
is already public, and is the wrong function to call here. Its own
docstring says why:

    Empty on any failure, which is the only safe answer here -- a
    caller cannot tell "no containers" from "the proxy is down" and
    must not act as though it could.

That erasure is the first link in run #83fc443e: proxy down ->
containers=[] -> the hint "forge-llm" matched nothing -> kernel logs
collected -> a confident, invented story about a variable blocking
searxng. It is a correct API for a caller that only needs names. It is
the exact thing this package exists to stop being the only option.

So this collector goes to the plumbing underneath instead --
harnais/host_exec.py's podman_cmd() and run_fixed(), which DO preserve
the distinction via the "[error] " prefix -- and turns it into an
Observation that can say "I could not look".

WHERE THAT PLUMBING LIVES

It used to live in graphs/sysadmin.py, and this file used to import
`_run_fixed` and `_DISCOVER_CONTAINERS_CMD` from it by their private
names. That was the lesser of two evils: rebuilding `podman ps` here
would have duplicated the SYSADMIN_PODMAN_URL wiring, the minimal
subprocess env, the timeout and the "[error] " convention -- four
things to drift instead of one, the cost `running_containers()`'s own
docstring names ("two readings of `podman ps` would drift the day the
command grows a flag").

The extraction that fixes it properly was deferred for one stated
reason -- sysadmin had to keep working untouched while the Context
Builder was built beside it -- and that reason ended when the Context
Builder was wired into the graph. It is now forge/harnais/host_exec.py,
which both this file and the graph import.

PRESENCE IS PROOF OF RUNNING

`podman ps` with no -a lists running containers only, so a name in
that output is an observation that the container was up -- the same
free fact sysadmin's _RUNNING_FACT already leans on. A container that
exists but is stopped is simply absent here, and this collector says
nothing about it rather than implying it does not exist.
"""

from datetime import datetime

from forge.config import (
    SYSADMIN_DISCOVERY_TIMEOUT,
    SYSADMIN_MAX_LOG_LINES,
)
from forge.harnais.collector import CostHint, Observation
from forge.harnais.facts import Fact
from forge.harnais.host_exec import NO_OUTPUT as _NO_OUTPUT
from forge.harnais.host_exec import podman_cmd
from forge.harnais.host_exec import run_fixed as _run_fixed

#: Imported rather than repeated. This literal was copied here and
#: into logs.py, and a third place spelled the same idea differently
#: and was wrong for it -- see graphs/sysadmin._collected_nothing.
#: It must never be parsed as a container named "[no output]", the same
#: class of mistake as the systemctl failure text that once became two
#: fake units called "System" and "Failed".

_ERROR_PREFIX = "[error]"

#: One line per container, tab-separated. NOT `--format json`, and the
#: reason is measured rather than stylistic: `_run_fixed` truncates its
#: output at SYSADMIN_MAX_LOG_LINES (100 in the deployed .env.local),
#: while `podman ps --format json` pretty-prints roughly twenty lines
#: per container. Five containers and the JSON is silently cut
#: mid-object -- a parse error at best, and at worst a shorter list
#: that looks complete. A template emits one line each, so the line cap
#: bounds the number of CONTAINERS reported rather than corrupting the
#: reply, and the count fact below is what makes that bound visible.
#:
#: `{{.Names}}` is the field this repo already runs in production. The
#: other two are not: if podman rejects either, it exits non-zero and
#: _run_fixed returns "[error] podman exited N: ..." carrying podman's
#: own message about the unknown field -- so a wrong guess here is a
#: failed Observation that names the mistake, never a silently wrong
#: number. That is the whole reason this is one command and not a
#: tolerant JSON parse: the error path already exists and it is better
#: than anything a parser could infer.
_PS_FORMAT = "{{.Names}}\t{{.StartedAt}}\t{{.Status}}"

_FIELDS = 3


def _containers_cmd() -> list[str]:
    """`podman ps` with the richer format, through the same proxy.

    This used to rebuild the `--url` wiring by hand, because it differs
    from harnais.host_exec.discover_containers_cmd() only in `--format`
    and there was no seam to pass that through. `podman_cmd(*args)` is
    that seam: how to reach podman is stated once, and this function
    says only what it wants from it.
    """
    return podman_cmd("ps", "--format", _PS_FORMAT)


class ContainersCollector:
    name = "containers"
    domains = ("container",)

    def is_available(self) -> bool:
        # Deliberately always True: "is podman reachable?" cannot be
        # answered without asking podman, and asking it IS collect().
        # A cheap pre-check here would either lie or run the command
        # twice, and collect() already reports the failure with the
        # real error text attached.
        return True

    def cost_hint(self) -> CostHint:
        return "moderate"

    def collect(self) -> Observation:
        at = datetime.now()  # noqa: DTZ005 -- local time, same as context_info
        raw = _run_fixed(_containers_cmd(), SYSADMIN_DISCOVERY_TIMEOUT)

        if raw.startswith(_ERROR_PREFIX):
            return Observation.failure(self.name, self.domains, raw, at)

        lines = (
            []
            if raw.strip() == _NO_OUTPUT
            else [line for line in raw.splitlines() if line.strip()]
        )

        if len(lines) >= SYSADMIN_MAX_LOG_LINES:
            # _run_fixed cuts at SYSADMIN_MAX_LOG_LINES and says nothing
            # about it, so a reply sitting exactly on the cap may or may
            # not be the whole list -- and `running_count` would state
            # the cap as an observed number. An under-count presented as
            # a fact is the same fault as an empty list presented as an
            # idle machine, so the run reports that it could not see the
            # whole list instead. It cannot fire below 100 containers.
            return Observation.failure(
                self.name,
                self.domains,
                f"[error] podman returned {len(lines)} lines, at or above "
                f"SYSADMIN_MAX_LOG_LINES ({SYSADMIN_MAX_LOG_LINES}): the "
                "container list may be truncated and its count would "
                "under-report",
                at,
            )

        try:
            rows = [_parse_row(line) for line in lines]
        except ValueError as e:
            # A shape this module did not expect is a failed
            # observation, not a shorter list. The message carries the
            # offending line so the first production run says what the
            # real format is instead of leaving someone to guess twice.
            return Observation.failure(self.name, self.domains, f"[error] {e}", at)

        # The count comes first and exists even when it is zero. That
        # is what makes "nothing is running" an observation rather than
        # an absence of one -- Observation refuses a successful run
        # with no facts precisely so this line cannot be forgotten.
        facts = [Fact("container", "running_count", len(rows), None, at, self.name)]
        for name, started_at, status in rows:
            # The entity goes in the key because the document's Fact
            # has no field for it (section 4). It makes `key` a
            # compound, which is tolerable at three collectors and
            # wants a real field once the Host Model in V2 gives
            # entities names of their own.
            facts.append(
                Fact("container", f"{name}.status", status, None, at, self.name)
            )
            facts.append(
                # THE POINT OF THIS COLLECTOR, and the reason it reads
                # more than names. llama-server fell over twice on
                # 2026-09-13 and `restart: unless-stopped` revived it
                # in silence -- a whole series of measurements crossed
                # a restart with nothing saying so. An uptime of 247
                # seconds next to a question about the last hour is
                # that fault, visible, with no history to consult and
                # no arithmetic for the model to get wrong.
                Fact(
                    "container",
                    f"{name}.uptime_s",
                    _uptime_s(started_at, at),
                    "s",
                    at,
                    self.name,
                )
            )
        return Observation.of(self.name, self.domains, facts, at)


def _parse_row(line: str) -> tuple[str, int, str]:
    """One `name<TAB>started_at<TAB>status` line, or ValueError.

    Strict on purpose. Tolerating a short row would mean emitting a
    container with no uptime, which reads exactly like a container that
    has been up forever -- absence wearing the shape of a measurement,
    which is the fault this whole package exists to remove.
    """
    parts = line.split("\t")
    if len(parts) != _FIELDS:
        raise ValueError(
            f"expected {_FIELDS} tab-separated fields from "
            f"`podman ps --format {_PS_FORMAT}`, got {len(parts)}: {line[:160]!r}"
        )
    name, started_at, status = (p.strip() for p in parts)
    if not name:
        raise ValueError(f"podman reported a container with no name: {line[:160]!r}")
    try:
        started = int(started_at)
    except ValueError:
        raise ValueError(
            f"expected a unix timestamp for {name!r}, got {started_at[:80]!r}"
        ) from None
    return name, started, status


def _uptime_s(started_at: int, now: datetime) -> int:
    """Seconds since the container started, floored at zero.

    Floored because a clock skew between the host podman reports from
    and this process would otherwise produce a negative uptime, which
    is not a fact about anything. Zero is the honest floor: it says
    "just now", which is what a skew of a few seconds means.
    """
    return max(0, int(now.timestamp()) - started_at)
