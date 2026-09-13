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

from forge.config import SYSADMIN_DISCOVERY_TIMEOUT
from forge.graphs.sysadmin import _DISCOVER_CONTAINERS_CMD, _run_fixed
from forge.harnais.collector import CostHint, Observation
from forge.harnais.facts import Fact

#: `_run_fixed` returns this literal when a command succeeded and
#: printed nothing. It must never be parsed as a container named
#: "[no output]" -- the same class of mistake as the systemctl failure
#: text that once became two fake units called "System" and "Failed".
_NO_OUTPUT = "[no output]"

_ERROR_PREFIX = "[error]"


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
        raw = _run_fixed(_DISCOVER_CONTAINERS_CMD(), SYSADMIN_DISCOVERY_TIMEOUT)

        if raw.startswith(_ERROR_PREFIX):
            return Observation.failure(self.name, self.domains, raw, at)

        names = (
            []
            if raw.strip() == _NO_OUTPUT
            else [line.strip() for line in raw.splitlines() if line.strip()]
        )

        # The count comes first and exists even when it is zero. That
        # is what makes "nothing is running" an observation rather than
        # an absence of one -- Observation refuses a successful run
        # with no facts precisely so this line cannot be forgotten.
        facts = [Fact("container", "running_count", len(names), None, at, self.name)]
        facts += [
            # The entity goes in the key because the document's Fact
            # has no field for it (section 4). It makes `key` a
            # compound, which is tolerable at three collectors and
            # wants a real field once the Host Model in V2 gives
            # entities names of their own.
            Fact("container", f"{name}.status", "running", None, at, self.name)
            for name in names
        ]
        return Observation.of(self.name, self.domains, facts, at)
