"""
Tests that dispatch actually goes through the Capability Registry, in
both execution paths (orchestrator and graph engine), and that the
one case the Registry cannot resolve is a hard stop rather than an
implicit choice.

That last case is unreachable in normal operation today -- every
capability has exactly one provider. It is tested anyway: the moment
a second provider appears, the difference between "fails loudly" and
"silently ran one of them" is the difference between noticing the
missing Cognitive Scheduler and shipping without it.
"""

import json

import pytest

import forge.orchestrator as orch_mod
from forge.errors import CapabilityAmbiguousError
from forge.kernel import registry as capabilities
from forge.kernel.capability import ToolCapability
from forge.orchestrator import Orchestrator
from forge.tools import registry as tool_registry


def _router_says(tool, content="x"):
    return lambda prompt: json.dumps({"tool": tool, "content": content})


def test_orchestrator_dispatches_through_a_capability(monkeypatch):
    monkeypatch.setitem(tool_registry.TOOLS, "chat", lambda c: "via capability")
    monkeypatch.setattr(orch_mod, "call_llm", _router_says("chat"))

    result = Orchestrator().run("hello")
    assert result.ok
    assert result.output == "via capability"


def test_orchestrator_refuses_to_choose_between_two_providers(monkeypatch):
    """
    Two providers, no Scheduler: the run must fail with a named
    architectural gap, not pick one and carry on.
    """
    monkeypatch.setitem(
        capabilities.REGISTERED,
        "chat",
        [
            ToolCapability(
                name="chat",
                provider="a_second_provider",
                handler=lambda c: "should never run",
                declared=True,
            )
        ],
    )
    monkeypatch.setattr(orch_mod, "call_llm", _router_says("chat"))

    result = Orchestrator().run("hello")
    assert not result.ok
    assert "no Cognitive Scheduler" in result.error
    assert "a_second_provider" in result.error


def test_ambiguity_error_is_typed():
    """
    Typed rather than a bare string, so the day it becomes reachable
    the orchestrator does not have to parse a message to know what
    happened -- same rule as the rest of forge.errors.
    """
    assert issubclass(CapabilityAmbiguousError, Exception)


def test_graph_engine_dispatches_through_a_capability(monkeypatch):
    from forge.graphs import default as graph_mod

    monkeypatch.setitem(tool_registry.TOOLS, "chat", lambda c: "graph via capability")
    monkeypatch.setattr(graph_mod, "call_llm", _router_says("chat"))

    state = graph_mod.build().run("hello")
    assert state.ok
    assert state.final_output == "graph via capability"


def test_graph_engine_refuses_to_choose_between_two_providers(monkeypatch):
    from forge.graphs import default as graph_mod

    monkeypatch.setitem(
        capabilities.REGISTERED,
        "chat",
        [
            ToolCapability(
                name="chat",
                provider="a_second_provider",
                handler=lambda c: "should never run",
                declared=True,
            )
        ],
    )
    monkeypatch.setattr(graph_mod, "call_llm", _router_says("chat"))

    state = graph_mod.build().run("hello")
    # The graph's fallback node deliberately resets ok=True and clears
    # the error so callers get a result rather than a crash, so the
    # property to pin here is that neither provider ran and the reason
    # reached the user, not that the state stayed failed.
    assert "no Cognitive Scheduler" in state.final_output
    assert "should never run" not in state.final_output


def test_a_capability_with_no_provider_passes_content_through(monkeypatch):
    """
    Preserved behaviour, not new: an unrouted decision returns its
    content rather than erroring. Pinned here so the registry swap
    cannot have changed it unnoticed.
    """
    monkeypatch.setattr(orch_mod, "call_llm", _router_says("chat"))
    monkeypatch.delitem(tool_registry.TOOLS, "chat", raising=False)

    result = Orchestrator().run("hello")
    assert result.ok
    assert result.output == "x"


@pytest.mark.parametrize("path", ["orchestrator", "graph"])
def test_both_paths_read_the_same_registry(monkeypatch, path):
    """
    The point of the swap: one place decides what a capability name
    resolves to, instead of two call sites each doing their own
    TOOLS lookup.
    """
    monkeypatch.setitem(tool_registry.TOOLS, "chat", lambda c: "shared source")

    if path == "orchestrator":
        monkeypatch.setattr(orch_mod, "call_llm", _router_says("chat"))
        assert Orchestrator().run("hi").output == "shared source"
    else:
        from forge.graphs import default as graph_mod

        monkeypatch.setattr(graph_mod, "call_llm", _router_says("chat"))
        assert graph_mod.build().run("hi").final_output == "shared source"
