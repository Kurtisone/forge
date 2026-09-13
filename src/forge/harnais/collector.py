"""
What one Collector run produced -- including the case where it produced
nothing, and why.

WHERE THIS DEPARTS FROM THE DESIGN DOCUMENT, AND WHY IT HAS TO

forge-kernel-harnais.md section 5 gives the Collector protocol as:

    def collect(self) -> list[Fact]: ...

and section 6 requires the MVP to produce, on a dead podman proxy,
"impossible d'observer l'état des containers (proxy indisponible)".

Those two cannot both hold. `list[Fact]` has exactly one way to say
"no facts" -- the empty list -- and a Fact cannot carry a failure,
because `Fact.confidence` is nailed to "observed". So a failed
collection and an idle machine arrive at the reader identically, which
is the bug the document is written to fix, one layer higher up.

This is not a hypothetical. graphs/sysadmin.py's `running_containers()`
already returns [] on any failure, and its docstring already names the
consequence: "a caller cannot tell 'no containers' from 'the proxy is
down' and must not act as though it could". Run #83fc443e is what
happens when a caller acts as though it could. Wrapping that same
erasure in a new type would have moved it, not fixed it.

So `collect()` returns an Observation: the facts, or the reason there
are none. That is the whole deviation.

THE INVARIANT THAT MAKES ABSENCE READABLE

    a successful Observation has at least one Fact
    a failed Observation has no Facts and a non-empty error

Which gives the Context Builder a rule with no third case: a domain
with no Fact behind it was NOT OBSERVED, full stop -- never "observed
to be empty". Collectors hold up their end by emitting a count fact
("0 containers running") rather than an empty list, so "nothing is
running" stays sayable as an observation.

`domains` exists for the same reason. A failed Observation carries no
Facts, so there is nothing in it to say WHAT went unobserved; the
collector has to declare that up front, before it knows whether it
will succeed.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from forge.harnais.facts import Fact

#: What a Collector costs to ask, for a Kernel deciding whether it is
#: worth asking (document section 3.2). Nothing reads it yet -- the MVP
#: asks all three collectors every time. It is declared rather than
#: measured, which is exactly why nothing is allowed to optimise on it:
#: see the three-phase rule in CLAUDE.md. Phase 2 measures it or it
#: goes away.
CostHint = Literal["cheap", "moderate", "expensive"]


@dataclass(frozen=True)
class Observation:
    """One Collector run. Either facts, or the reason there are none."""

    collector: str
    #: The domains this run speaks for, declared by the collector
    #: whether or not it succeeded.
    domains: tuple[str, ...]
    facts: tuple[Fact, ...]
    #: Empty string on success. Carries the command's own error text
    #: when it failed -- naming the command that broke is the single
    #: most useful thing a reader can be handed, the same reasoning as
    #: sysadmin's _collect_failed_node.
    error: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError(
                f"collector {self.collector!r} must declare the domains it "
                "speaks for, so a failure can say what went unobserved"
            )
        if self.error and self.facts:
            # Allowing both would reintroduce the third case the whole
            # invariant exists to remove: a domain that is partly
            # observed, which a reader has no way to weigh. A collector
            # that can half-fail reports a failure for the whole run --
            # see CpuRamCollector, which reads two files and fails on
            # either.
            raise ValueError(
                f"collector {self.collector!r} reported both facts and an "
                "error: a partial observation has no honest rendering"
            )
        if not self.error and not self.facts:
            # The trap this closes: a collector that "succeeded" with
            # nothing to show is indistinguishable from one that could
            # not run, which is precisely run #83fc443e. Emit a count
            # fact instead -- "0 containers running" is an observation.
            raise ValueError(
                f"collector {self.collector!r} returned no facts and no "
                "error: say what was observed to be absent, or say it "
                "could not be observed"
            )

    @property
    def failed(self) -> bool:
        return bool(self.error)

    @classmethod
    def of(
        cls, collector: str, domains: tuple[str, ...], facts: list[Fact], at: datetime
    ) -> "Observation":
        return cls(
            collector=collector,
            domains=domains,
            facts=tuple(facts),
            error="",
            observed_at=at,
        )

    @classmethod
    def failure(
        cls, collector: str, domains: tuple[str, ...], error: str, at: datetime
    ) -> "Observation":
        return cls(
            collector=collector,
            domains=domains,
            facts=(),
            error=error or "the collector failed without saying why",
            observed_at=at,
        )


@runtime_checkable
class Collector(Protocol):
    """
    Something that reads one part of the machine and says what it saw.

    It never reasons, never reads the user's question, and does not
    know an LLM exists (document section 2). It runs before anyone has
    read the question -- which is what makes its answer usable as
    evidence, and also why the Context Builder has to keep saying the
    facts may simply not cover what was asked.
    """

    name: str
    domains: tuple[str, ...]

    def is_available(self) -> bool: ...

    def collect(self) -> Observation: ...

    def cost_hint(self) -> CostHint: ...
