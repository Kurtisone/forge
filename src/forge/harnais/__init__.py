"""
The Harnais -- the only layer that touches the real machine.

It never reasons, never sees the user's question, and does not know an
LLM exists (document section 2). It turns raw readings into
timestamped `Fact`s with a named source, and says plainly when it
could not read something.

The MVP is observation only. There is no ActionCapability here and
there is not meant to be one until V3: what exists today proposes and
a human applies, the same posture graphs/sysadmin.py and tools/git.py
already hold.
"""

from forge.harnais.collector import Collector, CostHint, Observation
from forge.harnais.facts import Correlation, Fact, Hypothesis

__all__ = [
    "Collector",
    "Correlation",
    "CostHint",
    "Fact",
    "Hypothesis",
    "Observation",
    "default_collectors",
    "observe",
]


def default_collectors() -> list[Collector]:
    """
    The MVP collectors, in ascending cost order.

    Imported inside the function so that importing `forge.harnais` for
    its types alone does not pull in a subprocess-running module.
    Until the host plumbing moved to harnais/host_exec.py this was
    load-bearing in a stronger way -- the containers and logs
    collectors imported it from graphs.sysadmin, so `import
    forge.harnais` at module level would have dragged a whole graph
    in. It is now only about keeping facts.py cheap to import.
    """
    from forge.harnais.collectors.containers import ContainersCollector
    from forge.harnais.collectors.cpu_ram import CpuRamCollector
    from forge.harnais.collectors.logs import KernelLogsCollector
    from forge.harnais.collectors.units import SystemUnitsCollector

    return [
        CpuRamCollector(),
        ContainersCollector(),
        SystemUnitsCollector(),
        KernelLogsCollector(),
    ]


def observe(collectors: list[Collector]) -> list[Observation]:
    """
    Run every collector and return what each one saw -- or did not.

    A collector that declares itself unavailable still produces an
    Observation, a failed one. Skipping it silently would put its
    domains back in the state this package exists to remove: absent
    from the context with nothing saying why, which reads as "nothing
    to report".

    A collector that raises is caught for the same reason. One broken
    collector must not take the other two down with it, and a crash is
    just another way of not having observed something.
    """
    from datetime import datetime

    from forge.logger import log

    results: list[Observation] = []
    for collector in collectors:
        try:
            if not collector.is_available():
                results.append(
                    Observation.failure(
                        collector.name,
                        collector.domains,
                        "collector reports it is not available on this machine",
                        datetime.now(),  # noqa: DTZ005
                    )
                )
                continue
            results.append(collector.collect())
        except Exception as e:  # noqa: BLE001 - a crash is an unobserved domain
            log.warning("harnais: collector %r raised: %s", collector.name, e)
            results.append(
                Observation.failure(
                    collector.name,
                    collector.domains,
                    f"the collector raised {type(e).__name__}: {e}",
                    datetime.now(),  # noqa: DTZ005
                )
            )
    return results
