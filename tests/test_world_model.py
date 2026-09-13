"""
Tests for forge.kernel.world_model.

Three properties, in descending order of how much they matter:

1. The store can tell "observed to be zero" from "could not be read"
   from "never asked". That is the whole reason it holds anything
   beyond a dict of Facts.
2. A failed read drops what it can no longer vouch for, and records
   that it did -- nothing disappears silently.
3. The event window is bounded and stays ordered.
"""

from datetime import datetime, timedelta

from forge.harnais.collector import Observation
from forge.harnais.facts import Fact
from forge.kernel.world_model import (
    EVENT_OBSERVATION_FAILED,
    EVENT_STATE_CHANGED,
    InMemoryWorldModel,
    WorldEvent,
)

# Naive local time, as everywhere in Forge -- see context_info.py.
NOW = datetime.now()  # noqa: DTZ005


def fact(domain="container", key="running_count", value=0, at=None) -> Fact:
    return Fact(domain, key, value, None, at or NOW, "test")


def observed(*facts, collector="test", domains=("container",)) -> Observation:
    return Observation.of(collector, domains, list(facts), NOW)


def failed(error="[error] proxy down", domains=("container",)) -> Observation:
    return Observation.failure("test", domains, error, NOW)


# --- the three-way distinction ----------------------------------------------


def test_zero_containers_is_a_fact_not_an_absence():
    world = InMemoryWorldModel()
    world.record_observation(observed(fact(value=0)))

    assert [f.value for f in world.current_state("container")] == [0]
    assert world.observation_error("container") is None


def test_a_failed_read_leaves_no_facts_and_a_reason():
    world = InMemoryWorldModel()
    world.record_observation(failed())

    assert world.current_state("container") == []
    assert world.observation_error("container") == "[error] proxy down"


def test_a_domain_nobody_asked_about_has_neither():
    """
    The third case, and the one a naive store collapses into the
    second. None from observation_error does NOT mean healthy.
    """
    world = InMemoryWorldModel()

    assert world.current_state("gpu") == []
    assert world.observation_error("gpu") is None


def test_a_failure_marks_every_domain_the_collector_spoke_for():
    """cpu_ram covers two domains; when it fails, both go dark, and
    the failure carries no facts to derive them from."""
    world = InMemoryWorldModel()
    world.record_observation(failed(error="[error] /proc gone", domains=("cpu", "ram")))

    assert world.observation_error("cpu") == "[error] /proc gone"
    assert world.observation_error("ram") == "[error] /proc gone"


# --- what a failure does to what was known ----------------------------------


def test_a_failed_read_drops_the_domain_s_current_state():
    """
    `current_state` means "what is true now". After a failed read
    nothing is known to be true now -- keeping the old listing would
    let a context present a five-minute-old snapshot as the current
    one.
    """
    world = InMemoryWorldModel()
    world.record_observation(observed(fact("container", "forge-llm.status", "running")))
    assert world.current_state("container")

    world.record_observation(failed())
    assert world.current_state("container") == []


def test_the_drop_is_recorded_rather_than_silent():
    world = InMemoryWorldModel()
    world.record_observation(observed(fact("container", "forge-llm.status", "running")))
    world.record_observation(failed())

    events = world.recent_events(timedelta(minutes=5))
    assert [e.event_type for e in events] == [EVENT_OBSERVATION_FAILED]
    assert events[0].payload["dropped_facts"] == 1
    assert events[0].payload["error"] == "[error] proxy down"


def test_a_failure_in_one_domain_leaves_another_alone():
    world = InMemoryWorldModel()
    world.record_observation(
        observed(fact("ram", "used_pct", 40.2), collector="cpu_ram", domains=("ram",))
    )
    world.record_observation(failed())

    assert [f.key for f in world.current_state("ram")] == ["used_pct"]
    assert world.current_state("container") == []


def test_a_later_success_clears_an_earlier_failure():
    world = InMemoryWorldModel()
    world.record_observation(failed())
    world.record_observation(observed(fact(value=3)))

    assert world.observation_error("container") is None
    assert [f.value for f in world.current_state("container")] == [3]


# --- state and events -------------------------------------------------------


def test_a_changed_value_emits_a_state_changed_event():
    world = InMemoryWorldModel()
    world.record_fact(fact(value=3))
    world.record_fact(fact(value=2, at=NOW + timedelta(seconds=30)))

    events = world.recent_events(timedelta(minutes=5))
    assert [e.event_type for e in events] == [EVENT_STATE_CHANGED]
    assert events[0].payload["from"] == 3
    assert events[0].payload["to"] == 2


def test_an_unchanged_value_emits_nothing():
    """A change needs a before and an after that differ; re-reading the
    same number is not an event."""
    world = InMemoryWorldModel()
    world.record_fact(fact(value=3))
    world.record_fact(fact(value=3, at=NOW + timedelta(seconds=30)))

    assert world.recent_events(timedelta(minutes=5)) == []


def test_a_first_sighting_emits_nothing():
    world = InMemoryWorldModel()
    world.record_fact(fact(value=3))

    assert world.recent_events(timedelta(minutes=5)) == []


def test_current_state_keeps_the_most_recent_fact_per_key():
    world = InMemoryWorldModel()
    world.record_fact(fact("ram", "used_pct", 40.2))
    world.record_fact(fact("ram", "used_pct", 91.7, at=NOW + timedelta(seconds=30)))

    assert [f.value for f in world.current_state("ram")] == [91.7]


def test_current_state_is_ordered_by_key():
    """Deterministic order, so a context rendered twice from the same
    store is byte-identical -- which is what lets a KV prefix survive."""
    world = InMemoryWorldModel()
    for key in ("load_5m", "load_1m", "load_15m"):
        world.record_fact(fact("cpu", key, 1.0))

    assert [f.key for f in world.current_state("cpu")] == [
        "load_15m",
        "load_1m",
        "load_5m",
    ]


def test_recent_events_drops_what_fell_out_of_the_window():
    world = InMemoryWorldModel()
    world.record_event(WorldEvent("old", "cpu", {}, NOW - timedelta(hours=2)))
    world.record_event(WorldEvent("fresh", "cpu", {}, NOW))

    assert [e.event_type for e in world.recent_events(timedelta(minutes=5))] == [
        "fresh"
    ]


def test_recent_events_can_filter_by_domain():
    """The document's WorldEvent has no `domain` field even though its
    own protocol filters on one. The field was added; this is what it
    is for."""
    world = InMemoryWorldModel()
    world.record_event(WorldEvent("a", "cpu", {}, NOW))
    world.record_event(WorldEvent("b", "container", {}, NOW))

    assert [e.event_type for e in world.recent_events(timedelta(minutes=5), "cpu")] == [
        "a"
    ]


def test_the_event_window_is_bounded():
    world = InMemoryWorldModel(max_events=3)
    for n in range(10):
        world.record_event(WorldEvent(f"e{n}", "cpu", {}, NOW))

    events = world.recent_events(timedelta(minutes=5))
    assert [e.event_type for e in events] == ["e7", "e8", "e9"]


def test_domains_lists_both_the_observed_and_the_failed():
    world = InMemoryWorldModel()
    world.record_observation(
        observed(fact("ram", "used_pct", 40.2), collector="cpu_ram", domains=("ram",))
    )
    world.record_observation(failed())

    assert world.domains() == ["container", "ram"]


def test_nothing_is_persisted():
    """
    The MVP is in-memory by decision, not by omission (document section
    3.6 earmarks `memory_entries` for V2). Asserted so that adding
    persistence is a visible change rather than a silent one.
    """
    world = InMemoryWorldModel()
    world.record_fact(fact(value=3))

    assert InMemoryWorldModel().current_state("container") == []
