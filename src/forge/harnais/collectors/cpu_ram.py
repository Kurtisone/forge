"""
CPU load and memory, read straight from /proc.

WHAT THE DESIGN DOCUMENT GETS WRONG HERE

Section 9 describes all three MVP collectors as "wrapping des sources
déjà utilisées par sysadmin (busctl/proxy podman/journalctl)". That
mapping is off by one: busctl answers for systemd UNITS, not for CPU
or RAM, and nothing anywhere in Forge reads either today -- no
/proc/meminfo, no loadavg, no psutil, in src/, bench/ or deploy/.
So this collector is new plumbing, not a wrapper, and it is the only
one of the three that is.

Given that it had to be written, it is written without a subprocess:
two files, opened and parsed. That is cheaper than busctl, it is
exactly as read-only, and it adds no dependency -- which also means
this collector keeps working inside Forge's own container image, where
there is no journalctl, no podman and no bus (see config.py's note
above SYSADMIN_JOURNAL_DIR).

WHAT IT REPORTS INSIDE A CONTAINER

Whatever /proc says, which in a normal container is the HOST's memory
and the HOST's load, not the cgroup's. That is the right answer for
this machine -- the question is always "what is this box doing" -- but
it is a fact about the host, and `source` says "proc" rather than
"forge" so nobody has to guess whose numbers these are.

ALL OR NOTHING, ON PURPOSE

Either file failing fails the whole run. A half-read /proc is not a
case worth a code path, and Observation refuses partial results
anyway: a domain with no Fact behind it means NOT OBSERVED, and that
rule is only readable if it has no exceptions.

used_pct is arithmetic on two numbers read at the same instant from
the same file, not an interpretation, so it stays a Fact. It is here
because it is the line a human actually reads; MemTotal and
MemAvailable stay alongside it so the arithmetic is checkable.
"""

from datetime import datetime

from forge.harnais.collector import CostHint, Observation
from forge.harnais.facts import Fact

_MEMINFO = "/proc/meminfo"
_LOADAVG = "/proc/loadavg"

#: Load average windows, in the order /proc/loadavg prints them.
_LOAD_KEYS = ("load_1m", "load_5m", "load_15m")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _meminfo_kb(raw: str, field: str) -> int:
    """One `MemTotal:  15160368 kB` line, as an int. KeyError if the
    field is missing -- a /proc/meminfo without MemAvailable is a
    kernel older than 3.14 or not /proc at all, and guessing a value
    for it would be the exact move this package exists to prevent."""
    for line in raw.splitlines():
        name, _, rest = line.partition(":")
        if name == field:
            return int(rest.split()[0])
    raise KeyError(f"{field} not found in {_MEMINFO}")


class CpuRamCollector:
    name = "cpu_ram"
    domains = ("cpu", "ram")

    def is_available(self) -> bool:
        try:
            _read(_MEMINFO)
            _read(_LOADAVG)
        except OSError:
            return False
        return True

    def cost_hint(self) -> CostHint:
        return "cheap"

    def collect(self) -> Observation:
        at = datetime.now()  # noqa: DTZ005 -- local time, same as context_info
        try:
            meminfo = _read(_MEMINFO)
            loadavg = _read(_LOADAVG)
            total = _meminfo_kb(meminfo, "MemTotal")
            available = _meminfo_kb(meminfo, "MemAvailable")
            loads = [float(v) for v in loadavg.split()[:3]]
        except (OSError, KeyError, ValueError, IndexError) as e:
            return Observation.failure(
                self.name,
                self.domains,
                f"cannot read {_MEMINFO}/{_LOADAVG}: {e}",
                at,
            )

        if total <= 0 or len(loads) < len(_LOAD_KEYS):
            return Observation.failure(
                self.name,
                self.domains,
                f"{_MEMINFO}/{_LOADAVG} parsed but made no sense "
                f"(MemTotal={total}, {len(loads)} load values)",
                at,
            )

        used_pct = round((total - available) * 100.0 / total, 1)
        facts = [
            Fact("ram", "total_kb", total, "kB", at, self.name),
            Fact("ram", "available_kb", available, "kB", at, self.name),
            Fact("ram", "used_pct", used_pct, "%", at, self.name),
        ]
        facts += [
            Fact("cpu", key, load, None, at, self.name)
            for key, load in zip(_LOAD_KEYS, loads, strict=False)
        ]
        return Observation.of(self.name, self.domains, facts, at)
