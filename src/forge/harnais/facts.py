"""
The three things this system is allowed to believe, kept apart by type.

ARCHITECTURE of the distinction (forge-kernel-harnais.md, section 1):

    FACT        observed directly by the Harnais, timestamped, with a
                traceable source
    CORRELATION two facts close in time, computed by the Kernel --
                never by the LLM
    HYPOTHESIS  produced from facts/correlations by something that
                reasons, never persisted as truth

WHY THIS IS A TYPE AND NOT A PROMPT RULE

Run #83fc443e: the podman proxy was down, `podman ps` returned
nothing, the caller could not tell "no containers" from "cannot ask",
and the model was handed kernel logs and a question about a named
container. It produced a confident, entirely invented story. Every
step was individually reasonable.

Three later runs said the same thing a different way (#7a29f59d,
#7e0ed90c, #be385d16): given evidence about a different subject, or a
403, or a routine notice, this model answers fluently anyway. Nine
wording fixes have lost on this codebase against that exact failure.

So the separation is not a sentence in a prompt asking for honesty.
`Fact` cannot be constructed with a confidence other than "observed",
`Hypothesis` cannot omit who produced it, and `Correlation` has no
field in which to name a cause. What the model writes can only ever
become a `Hypothesis`, and a `Hypothesis` is visibly not a `Fact` at
every point downstream.

WHAT `confidence` BEING init=False BUYS

    Fact(..., confidence="high")   -> TypeError, at construction
    fact.confidence = "high"       -> FrozenInstanceError
    replace(fact, confidence=...)  -> refused by dataclasses

It is still a declared field, so it survives asdict(), repr() and
equality -- a Fact serialized into a log or a context line carries its
own claim to being observed. The one remaining door is
`object.__setattr__`, which no accident opens: it is a deliberate act
of subversion, not a code path anyone reaches by writing ordinary
code. `__post_init__` closes the other one -- a subclass redefining
the default -- because that one IS reachable by writing ordinary code.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class Fact:
    """
    One thing the Harnais saw, at one moment, from one named source.

    `value` is deliberately untyped: a load average is a float, a
    container status a string, a log tail a multi-line block. What
    makes it a Fact is not its Python type but that something read it
    off the machine and can say where.

    `domain`/`key` are the World Model's primary key (see
    kernel/world_model.py). The document's Fact has no field for the
    ENTITY a fact is about, so per-entity facts encode it in `key`
    ("forge-llm.status"). That is a compound key and it is a known
    weakness -- see the note in collectors/containers.py.
    """

    domain: str  # "cpu", "ram", "container", "logs", ...
    key: str  # "usage_pct", "forge-llm.status", "journalctl -k.tail"
    value: Any
    unit: str | None
    observed_at: datetime
    source: str  # the Collector's name, for traceability

    #: Never a parameter. See the module docstring.
    confidence: Literal["observed"] = field(init=False, default="observed")

    def __post_init__(self) -> None:
        # Reachable only from a subclass that redefines the default.
        # A Fact that is not observed is the bug this module exists
        # for, so it fails loudly rather than flowing downstream.
        if self.confidence != "observed":
            raise ValueError(
                f"a Fact is observed by definition, not {self.confidence!r}"
            )
        # An untraceable fact is a rumour with a timestamp. The whole
        # value of this layer is that a reader can ask "who says so?"
        # and get an answer, so the empty answer is refused here
        # rather than rendered as "(source: )" three modules later.
        if not self.source.strip():
            raise ValueError("a Fact must name the collector that observed it")


@dataclass(frozen=True)
class Correlation:
    """
    Two or more facts the KERNEL placed side by side, and nothing more.

    There is no `cause` field and no `effect` field, on purpose: the
    shape itself cannot express causality, so no amount of downstream
    formatting can quietly promote a coincidence into an explanation.
    `facts` is an unordered tuple -- a set of things that co-occurred,
    not an arrow between them.

    Nothing in the MVP produces one of these yet; correlation arrives
    in V2 with the Host Model's dependency edges, which is what would
    make "temporal proximity" mean more than "at the same time".
    """

    facts: tuple[Fact, ...]
    relation: str  # "temporal_proximity", "dependency_chain", ...
    computed_at: datetime
    strength: Literal["weak", "moderate", "strong"]

    def __post_init__(self) -> None:
        # One fact next to itself is not a correlation; it is a fact
        # with extra packaging, and it would render as though two
        # independent observations agreed.
        if len(self.facts) < 2:
            raise ValueError("a Correlation relates at least two Facts")


@dataclass(frozen=True)
class Hypothesis:
    """
    A statement nobody observed, carrying who said it and on what.

    This is the only container in the codebase for a claim that is not
    backed by an observation, and it is designed to be impossible to
    mistake for one: `produced_by` is mandatory ("llm:qwen3.8-9b"), and
    a hypothesis resting on nothing cannot claim better than "low".

    That last rule is the one with teeth. The failure this whole branch
    exists to stop is not a model being wrong -- it is a model being
    CONFIDENT while its evidence is empty. A confidence level is a
    choice the producer would otherwise make freely; when `based_on` is
    empty the choice is removed rather than requested.
    """

    statement: str
    based_on: tuple[Fact | Correlation, ...]
    confidence: Literal["low", "medium", "high"]
    produced_by: str  # "llm:qwen3.8-9b", never confusable with a Fact
    produced_at: datetime

    def __post_init__(self) -> None:
        if not self.produced_by.strip():
            raise ValueError("a Hypothesis must name what produced it")
        if not self.based_on and self.confidence != "low":
            raise ValueError(
                f"a Hypothesis based on no observation cannot be "
                f"{self.confidence!r} confidence"
            )
