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

So this collector reaches past it to the plumbing underneath --
_DISCOVER_CONTAINERS_CMD() and _run_fixed(), which DO preserve the
distinction via the "[error] " prefix -- and turns it into an
Observation that can say "I could not look".

WHY PRIVATE NAMES, DELIBERATELY

Importing `_`-prefixed names across modules is a smell, and the two
alternatives are worse. Rebuilding `podman ps --format {{.Names}}`
here duplicates the command, and `running_containers()`'s docstring
already names that cost ("Two readings of `podman ps` would drift the
day the command grows a flag") -- it would also duplicate the
SYSADMIN_PODMAN_URL wiring, the minimal subprocess env, the timeout
and the "[error] " convention, which is four places to drift instead
of one. Reaching in keeps a single definition of what talking to
podman means.

The tidy version of this is to lift _run_fixed/_subprocess_env and the
command builders out of graphs/sysadmin.py into a shared module both
import. That is a pure extraction with no behaviour change, and it is
deliberately NOT done on this branch: sysadmin keeps working untouched
until the Context Builder is actually wired in.

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
    SYSADMIN_PODMAN_URL,
)
from forge.graphs.sysadmin import _run_fixed
from forge.harnais.collector import CostHint, Observation
from forge.harnais.facts import Fact

#: `_run_fixed` returns this literal when a command succeeded and
#: printed nothing. It must never be parsed as a container named
#: "[no output]" -- the same class of mistake as the systemctl failure
#: text that once became two fake units called "System" and "Failed".
_NO_OUTPUT = "[no output]"

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

    Mirrors graphs.sysadmin._DISCOVER_CONTAINERS_CMD's three-line shape
    rather than importing it, because only the `--format` argument
    differs and there is no seam to pass it through. That is a real
    duplication of the SYSADMIN_PODMAN_URL wiring, and the note at the
    top of this module about lifting the plumbing into a shared module
    is now the fix for two callers instead of one.
    """
    base = ["podman"]
    if SYSADMIN_PODMAN_URL:
        base += ["--url", SYSADMIN_PODMAN_URL]
    return base + ["ps", "--format", _PS_FORMAT]


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
