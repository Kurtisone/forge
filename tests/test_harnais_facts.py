"""
Tests for forge.harnais.facts -- the wall between observed and inferred.

The property under test is not "these dataclasses hold their fields".
It is that the wall has no door: there is no way, from inside ordinary
Python, to end up holding a `Fact` that nobody observed, or a confident
`Hypothesis` resting on nothing.

Asserted by trying to build the bad object, not by reading the type
annotations. `Literal["observed"]` is documentation at runtime --
nothing enforces it -- so a test that only checked the annotation would
pass against a class anyone could construct with confidence="high".
"""

import dataclasses
from datetime import datetime

import pytest

from forge.harnais.facts import Correlation, Fact, Hypothesis

# Naive local time, as everywhere in Forge -- see context_info.py.
NOW = datetime(2026, 9, 13, 17, 40, 12)  # noqa: DTZ001


def a_fact(domain: str = "ram", key: str = "used_pct", value=40.2) -> Fact:
    return Fact(domain, key, value, "%", NOW, "cpu_ram")


# --- Fact -------------------------------------------------------------------


def test_a_fact_is_observed_and_says_who_by():
    fact = a_fact()
    assert fact.confidence == "observed"
    assert fact.source == "cpu_ram"
    assert fact.observed_at == NOW


def test_confidence_is_not_a_constructor_parameter():
    """
    The central claim of this module, asserted the only way that means
    anything: by trying it.
    """
    with pytest.raises(TypeError):
        Fact("ram", "used_pct", 40.2, "%", NOW, "cpu_ram", confidence="high")


def test_confidence_cannot_be_reassigned():
    with pytest.raises(dataclasses.FrozenInstanceError):
        a_fact().confidence = "high"


def test_confidence_cannot_be_replaced():
    """
    Both exception types on purpose: dataclasses.replace() raises
    ValueError for an init=False field on Python 3.12 (what CI runs)
    and TypeError on 3.13 (what this machine runs). Pinning either one
    would make the test fail on the other interpreter for a reason that
    has nothing to do with what it checks.
    """
    with pytest.raises((TypeError, ValueError), match="init=False"):
        dataclasses.replace(a_fact(), confidence="low")


def test_replacing_any_other_field_keeps_the_fact_observed():
    """`replace` stays usable -- it is the confidence that is nailed
    down, not the dataclass."""
    assert dataclasses.replace(a_fact(), value=41.0).confidence == "observed"


def test_a_subclass_cannot_redefine_confidence():
    """
    The one remaining route a programmer reaches by accident: inherit
    and override the default. __post_init__ closes it.
    """
    with pytest.raises(ValueError, match="observed by definition"):
        dataclasses.make_dataclass(
            "GuessedFact",
            [("confidence", str, dataclasses.field(default="high"))],
            bases=(Fact,),
            frozen=True,
        )("ram", "used_pct", 40.2, "%", NOW, "cpu_ram")


def test_a_fact_must_name_its_source():
    """An untraceable fact is a rumour with a timestamp."""
    with pytest.raises(ValueError, match="must name the collector"):
        Fact("ram", "used_pct", 40.2, "%", NOW, "   ")


def test_confidence_survives_serialization():
    """
    init=False rather than a property, so a Fact written to a log or a
    trace still carries its own claim to being observed.
    """
    assert dataclasses.asdict(a_fact())["confidence"] == "observed"


# --- Hypothesis -------------------------------------------------------------


def test_a_hypothesis_resting_on_nothing_cannot_be_confident():
    """
    The failure this branch exists for is not a model being wrong. It
    is a model being CONFIDENT with no evidence -- so when the evidence
    is empty, the choice of confidence is removed rather than asked for.
    """
    for claimed in ("medium", "high"):
        with pytest.raises(ValueError, match="cannot be"):
            Hypothesis("le conteneur plante", (), claimed, "llm:qwen3.8-9b", NOW)


def test_a_hypothesis_resting_on_nothing_may_still_be_stated_as_low():
    hypothesis = Hypothesis("le conteneur plante", (), "low", "llm:qwen3.8-9b", NOW)
    assert hypothesis.confidence == "low"
    assert hypothesis.based_on == ()


def test_a_hypothesis_backed_by_facts_may_claim_more():
    hypothesis = Hypothesis("la RAM sature", (a_fact(),), "high", "llm:qwen3.8-9b", NOW)
    assert hypothesis.confidence == "high"


def test_a_hypothesis_must_name_its_producer():
    """'llm:qwen3.8-9b' is what stops a hypothesis being read back as
    an observation three modules later."""
    with pytest.raises(ValueError, match="must name what produced it"):
        Hypothesis("la RAM sature", (a_fact(),), "high", "", NOW)


def test_a_hypothesis_is_not_a_fact():
    """Stated as a type assertion because every downstream renderer
    branches on exactly this."""
    hypothesis = Hypothesis("la RAM sature", (a_fact(),), "low", "llm:x", NOW)
    assert not isinstance(hypothesis, Fact)
    assert hypothesis.confidence != "observed"


# --- Correlation ------------------------------------------------------------


def test_a_correlation_needs_two_facts():
    """One fact beside itself is a fact with packaging, and would read
    as two independent observations agreeing."""
    with pytest.raises(ValueError, match="at least two"):
        Correlation((a_fact(),), "temporal_proximity", NOW, "weak")


def test_a_correlation_has_nowhere_to_put_a_cause():
    """
    Section 8's named risk, closed by shape rather than by review: the
    dataclass has no cause/effect field, so no downstream formatting
    can promote a coincidence into an explanation.
    """
    fields = {f.name for f in dataclasses.fields(Correlation)}
    assert not fields & {"cause", "effect", "because", "leads_to"}
    assert fields == {"facts", "relation", "computed_at", "strength"}


def test_a_correlation_relates_facts_only():
    correlation = Correlation(
        (a_fact(), a_fact("cpu", "load_1m", 3.2)), "temporal_proximity", NOW, "weak"
    )
    assert all(f.confidence == "observed" for f in correlation.facts)
