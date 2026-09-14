"""
Recent kernel log lines, via the journalctl sysadmin already uses.

NO TARGET PARAMETER, AND THAT IS THE POINT

This collector runs `journalctl -k` and nothing else. It takes no unit
name, has no slot to put one in, and therefore cannot be the path by
which a name the router invented reaches a subprocess.

graphs/sysadmin.py spends its whole security model on that one risk:
a target only reaches `journalctl --unit=<name>` after appearing
verbatim in the same run's own discovery output, and
tests/test_sysadmin.py patches subprocess.run itself rather than the
_run_fixed boundary so a regression cannot hide behind a mock. A
`LogsCollector(unit=...)` would reopen that door and defend it with a
docstring asking callers to validate first -- a rule written in prose,
which is the shape this codebase has watched fail thirteen times.

Per-unit logs therefore arrive with the validator that makes them
safe, not before. That is a real gap in this MVP and it is named as
one: kernel logs answer "what is wrong with this box", never "why did
forge-llm restart".

WHAT A LOG FACT IS

Two facts, and the count comes first:

    logs / journalctl -k.lines = 12
    logs / journalctl -k.tail  = <the block>

"journalctl -k returned exactly this text at 17:40:12" is observed,
timestamped and traceable, so the block is a legitimate Fact value
even though it is prose rather than a number. The count exists so that
an EMPTY journal is still an observation -- `journalctl` on a window
with no entries and a journal nobody can read look identical from
here, and only one of them is a fact about the machine.

An empty log block is not a quiet system. The Context Builder says so
in those words, for the same reason sysadmin's _nothing_collected_node
does: measured on 2026-09-12, asked whether an EMPTY log block
answered "pourquoi searxng a redémarré ?", this model said yes.
"""

from datetime import datetime

from forge.config import SYSADMIN_COLLECT_TIMEOUT
from forge.graphs.sysadmin import NO_OUTPUT as _NO_OUTPUT
from forge.graphs.sysadmin import _collect_cmd, _run_fixed
from forge.harnais.collector import CostHint, Observation
from forge.harnais.facts import Fact

#: What `journalctl -k` is called when a fact names its own source.
#: Matches the string sysadmin puts in `log_source`, so the two read
#: the same in a trace.
_SOURCE = "journalctl -k"

_ERROR_PREFIX = "[error]"


class KernelLogsCollector:
    name = "logs"
    domains = ("logs",)

    def is_available(self) -> bool:
        # Same reasoning as ContainersCollector: whether journalctl can
        # reach a journal is only answerable by running it, and
        # collect() reports that failure with the real error text.
        return True

    def cost_hint(self) -> CostHint:
        return "moderate"

    def collect(self) -> Observation:
        at = datetime.now()  # noqa: DTZ005 -- local time, same as context_info
        raw = _run_fixed(_collect_cmd("kernel", ""), SYSADMIN_COLLECT_TIMEOUT)

        if raw.startswith(_ERROR_PREFIX):
            return Observation.failure(self.name, self.domains, raw, at)

        block = "" if raw.strip() == _NO_OUTPUT else raw.strip()
        lines = block.splitlines() if block else []

        facts = [Fact("logs", f"{_SOURCE}.lines", len(lines), "lines", at, self.name)]
        if lines:
            facts.append(Fact("logs", f"{_SOURCE}.tail", block, None, at, self.name))
        return Observation.of(self.name, self.domains, facts, at)
