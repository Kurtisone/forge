"""
What is true right now, what changed recently, and what could not be read.

The document (section 3.5) asks for an in-memory store with three
parts: current state per metric, a bounded window of recent events,
and later the correlations between them. Correlation waits for V2 --
it needs the Host Model's dependency edges to mean more than "two
things happened at the same time", and a `Correlation` computed from
temporal proximity alone is the shape of the mistake section 8 warns
about.

THE PART THAT IS NOT IN THE DOCUMENT, AND HAS TO BE

A store of Facts can only ever answer "here is what I have". The
question the Context Builder actually needs answered is the negative
one: "is there nothing here because nothing is happening, or because
nobody could look?" Facts cannot carry that -- `Fact.confidence` is
nailed to "observed" -- so the store tracks it alongside them, fed by
`record_observation`. See harnais/collector.py for the full argument.

Hence the rule:

    a domain with Facts        -> observed, state them
    a domain with a failure    -> NOT observed, say why
    a domain with neither      -> never asked, say that

WHY A FAILED READ DROPS THE DOMAIN'S CURRENT STATE

`current_state` means "what is true now". After a failed read, nothing
is known to be true now -- only what WAS true at the last successful
one. Keeping the old Facts would let the Context Builder present a
five-minute-old container listing as the current one, which is the
same confabulation as before with a smaller lie in it.

The cost is real: a RAM reading five seconds old is genuinely useful
and this throws it away. The alternative is a third rendering case
("stale fact, last read failed"), and three cases is how a reader
stops reading. The drop is recorded as an `observation_failed` event
rather than done silently, so the history still shows what happened,
and V2 -- which persists events and can mark staleness properly -- is
where the better answer belongs.

NO NEW CONFIGURATION

`max_events` is a constructor argument with a module default, not an
env var. Nothing on this branch reads from or writes to the deployed
.env.local, so nothing about the running deployment changes when it
lands.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from forge.harnais.collector import Observation
from forge.harnais.facts import Fact

#: How many events the window holds. Roughly an hour of the three MVP
#: collectors changing every value they have, every minute -- enough
#: to answer "what happened just before this", which is the only
#: question the window exists for. Not an env var; see the module
#: docstring.
DEFAULT_MAX_EVENTS = 512

#: Emitted when a (domain, key) that had a value gets a different one.
#: This is the `StateChanged` of document section 3.3, computed here
#: rather than by the collectors -- a collector reports what it sees,
#: it has no memory of what it saw last time.
EVENT_STATE_CHANGED = "state_changed"

#: Emitted when an observation fails, carrying which domains went dark
#: and how many known facts were dropped as a result.
EVENT_OBSERVATION_FAILED = "observation_failed"


@dataclass(frozen=True)
class WorldEvent:
    """
    Something that happened, as opposed to something that is.

    Two deliberate departures from document section 4:

    - `domain`, which the document's WorldEvent does not have even
      though section 5 asks `recent_events(window, domain=...)` to
      filter on it. The signature and the shape could not both be
      right; the shape gained a field.
    - frozen, and `related_nodes` a tuple. The document leaves this
      one mutable. An audit record that can be edited after the fact
      is not an audit record, and section 3.11 wants every fact,
      decision and action on the same journal.
    """

    event_type: str
    domain: str
    payload: dict[str, Any]
    timestamp: datetime
    related_nodes: tuple[str, ...] = ()


class WorldModelStore(Protocol):
    """Document section 5, plus the two methods absence needs."""

    def record_fact(self, fact: Fact) -> None: ...

    def record_event(self, event: WorldEvent) -> None: ...

    def recent_events(
        self, window: timedelta, domain: str | None = None
    ) -> list[WorldEvent]: ...

    def current_state(self, domain: str) -> list[Fact]: ...

    def record_observation(self, observation: Observation) -> None: ...

    def observation_error(self, domain: str) -> str | None: ...


@dataclass
class InMemoryWorldModel:
    """
    The MVP store: a dict, a bounded deque, and a failure register.

    No persistence. Section 3.6 earmarks `memory_entries` with a
    `world_event` type for that, and it is V2 work: persisting a model
    whose shape is three days old would commit a schema before anything
    has read it in anger.
    """

    max_events: int = DEFAULT_MAX_EVENTS

    #: (domain, key) -> the most recent Fact for it.
    _state: dict[tuple[str, str], Fact] = field(default_factory=dict)
    _events: deque[WorldEvent] = field(default_factory=deque)
    #: domain -> why its last observation produced nothing.
    _failures: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._events = deque(self._events, maxlen=self.max_events)

    # --- writing ----------------------------------------------------------

    def record_fact(self, fact: Fact) -> None:
        """
        Store a fact as the current value of its (domain, key), and
        emit a state_changed event if it displaced a different one.

        Recording a fact is proof its domain was observable, so it
        clears any failure standing against that domain. Order matters
        the other way round too -- see record_observation, which fails
        a domain only after deciding it has no facts to record.
        """
        slot = (fact.domain, fact.key)
        previous = self._state.get(slot)
        self._state[slot] = fact
        self._failures.pop(fact.domain, None)

        if previous is not None and previous.value != fact.value:
            self.record_event(
                WorldEvent(
                    event_type=EVENT_STATE_CHANGED,
                    domain=fact.domain,
                    payload={
                        "key": fact.key,
                        "from": previous.value,
                        "to": fact.value,
                        "unit": fact.unit,
                        "source": fact.source,
                    },
                    timestamp=fact.observed_at,
                )
            )

    def record_event(self, event: WorldEvent) -> None:
        self._events.append(event)

    def record_observation(self, observation: Observation) -> None:
        """
        Take in one Collector run, facts and absence alike.

        This is the only door between the Harnais and the Kernel. A
        successful run records its facts; a failed one marks every
        domain it spoke for as unobserved and drops what was known
        about them, because "what is true now" is no longer answerable
        for those domains.
        """
        if not observation.failed:
            for fact in observation.facts:
                self.record_fact(fact)
            return

        dropped = 0
        for domain in observation.domains:
            self._failures[domain] = observation.error
            for slot in [s for s in self._state if s[0] == domain]:
                del self._state[slot]
                dropped += 1

        self.record_event(
            WorldEvent(
                event_type=EVENT_OBSERVATION_FAILED,
                domain=observation.domains[0],
                payload={
                    "collector": observation.collector,
                    "domains": list(observation.domains),
                    "error": observation.error,
                    "dropped_facts": dropped,
                },
                timestamp=observation.observed_at,
            )
        )

    # --- reading ----------------------------------------------------------

    def current_state(self, domain: str) -> list[Fact]:
        """Every fact currently known about `domain`, in key order."""
        return [fact for slot, fact in sorted(self._state.items()) if slot[0] == domain]

    def observation_error(self, domain: str) -> str | None:
        """
        Why `domain` has no facts, or None.

        None does NOT mean "fine": a domain nobody ever asked about has
        no facts and no failure either. The caller has to weigh this
        against current_state() -- see context_builder._render_domain,
        which is the one place that does.
        """
        return self._failures.get(domain)

    def recent_events(
        self, window: timedelta, domain: str | None = None
    ) -> list[WorldEvent]:
        """Events inside `window`, oldest first."""
        cutoff = datetime.now() - window  # noqa: DTZ005 -- local, as observed_at
        return [
            event
            for event in self._events
            if event.timestamp >= cutoff and (domain is None or event.domain == domain)
        ]

    def domains(self) -> list[str]:
        """Every domain this store has heard about, observed or failed."""
        return sorted({slot[0] for slot in self._state} | set(self._failures))
