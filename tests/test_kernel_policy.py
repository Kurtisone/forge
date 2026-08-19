"""
Tests for forge.kernel.policy: the deterministic deny gate, and its
integration into both execution paths.

The policy flags are read into forge.kernel.policy at import time, so
tests monkeypatch them there rather than on forge.config -- the same
pattern as tests/test_registry.py with ENABLED_TOOLS.

Three properties are asserted rather than merely documented:

1. The gate only ever SUBTRACTS. No combination of flags can make a
   capability run that was not already reachable, and a capability
   requiring nothing is allowed no matter what is switched off.
2. Denials are explained. A blocked capability returns a reason that
   reaches the caller, because a capability that silently does not run
   is indistinguishable from one that ran badly.
3. Denial is by declared requirement, never by tool name -- so a tool
   that declares nothing is caught by every flag, and a new tool is
   covered the day it lands without editing a list here.
"""

import json

import pytest

from forge.kernel import policy
from forge.kernel.capability import LOCAL_READONLY, Requirements, ToolCapability
from forge.orchestrator import Orchestrator

NETWORK_ONLY = Requirements(
    network=True, llm=False, mutates_workspace=False, spawns_process=False
)
WRITES_ONLY = Requirements(
    network=False, llm=False, mutates_workspace=True, spawns_process=False
)
SUBPROCESS_ONLY = Requirements(
    network=False, llm=False, mutates_workspace=False, spawns_process=True
)
LLM_ONLY = Requirements(
    network=False, llm=True, mutates_workspace=False, spawns_process=False
)


def _cap(name: str, requirements: Requirements) -> ToolCapability:
    return ToolCapability(
        name=name,
        provider=name,
        handler=lambda content: content,
        requirements=requirements,
        declared=True,
    )


def _deny(monkeypatch, **flags) -> None:
    """Switch policy flags off by name, e.g. _deny(mp, network=False)."""
    mapping = {
        "network": "POLICY_ALLOW_NETWORK",
        "writes": "POLICY_ALLOW_WORKSPACE_WRITES",
        "subprocess": "POLICY_ALLOW_SUBPROCESS",
    }
    for key, value in flags.items():
        monkeypatch.setattr(policy, mapping[key], value)


# --- Verdict ----------------------------------------------------------------


def test_verdict_is_truthy_when_allowed():
    assert policy.Verdict(allowed=True)
    assert not policy.Verdict(allowed=False, reason="nope")


def test_allowed_verdict_carries_no_reason():
    assert policy.check(_cap("chat", LOCAL_READONLY)).reason == ""


# --- Default posture --------------------------------------------------------


def test_everything_is_allowed_by_default():
    """An untouched deployment must behave as if this module did not exist."""
    for requirements in (LOCAL_READONLY, NETWORK_ONLY, WRITES_ONLY, SUBPROCESS_ONLY):
        assert policy.check(_cap("x", requirements)).allowed is True


# --- Denial by declared requirement -----------------------------------------


@pytest.mark.parametrize(
    ("flag", "requirements", "expected"),
    [
        ("network", NETWORK_ONLY, "network access"),
        ("writes", WRITES_ONLY, "workspace writes"),
        ("subprocess", SUBPROCESS_ONLY, "subprocesses"),
    ],
)
def test_each_flag_denies_its_own_requirement(
    monkeypatch, flag, requirements, expected
):
    _deny(monkeypatch, **{flag: False})
    verdict = policy.check(_cap("target", requirements))
    assert verdict.allowed is False
    assert expected in verdict.reason
    assert "target" in verdict.reason


def test_a_flag_does_not_deny_an_unrelated_requirement(monkeypatch):
    _deny(monkeypatch, network=False)
    assert policy.check(_cap("files", WRITES_ONLY)).allowed is True
    assert policy.check(_cap("test", SUBPROCESS_ONLY)).allowed is True


def test_local_readonly_survives_every_flag_being_off(monkeypatch):
    """
    The gate only subtracts: a capability that requires nothing has
    nothing to subtract, whatever the context.
    """
    _deny(monkeypatch, network=False, writes=False, subprocess=False)
    assert policy.check(_cap("chat", LOCAL_READONLY)).allowed is True


def test_llm_alone_is_never_denied(monkeypatch):
    """
    There is deliberately no POLICY_ALLOW_LLM. The router itself calls
    the LLM on every single turn, so a flag denying LLM use would deny
    Forge the ability to route at all -- the useful question is *which*
    model runs where, which belongs to the Scheduler, not to a boolean
    gate. Asserted so the omission reads as a decision, not an oversight.
    """
    _deny(monkeypatch, network=False, writes=False, subprocess=False)
    assert policy.check(_cap("review", LLM_ONLY)).allowed is True


def test_several_blocked_requirements_are_all_named(monkeypatch):
    _deny(monkeypatch, network=False, writes=False, subprocess=False)
    verdict = policy.check(_cap("shell", Requirements()))
    assert verdict.allowed is False
    for expected in ("network access", "workspace writes", "subprocesses"):
        assert expected in verdict.reason


def test_the_reason_order_is_stable(monkeypatch):
    """
    Reasons are built in declaration order, not in the order flags were
    flipped, so the same capability always explains itself the same way.
    """
    _deny(monkeypatch, network=False, subprocess=False)
    first = policy.check(_cap("shell", Requirements())).reason
    _deny(monkeypatch, subprocess=False, network=False)
    assert policy.check(_cap("shell", Requirements())).reason == first


def test_an_undeclared_capability_is_caught_by_every_flag(monkeypatch):
    """
    Fail-closed, end to end: undeclared requirements default to the most
    demanding profile, so a tool that declares nothing is denied by any
    restriction rather than slipping through it.
    """
    undeclared = ToolCapability(name="mystery", provider="mystery", handler=lambda c: c)
    assert undeclared.declared is False
    for flag in ("network", "writes", "subprocess"):
        mp = pytest.MonkeyPatch()
        _deny(mp, **{flag: False})
        assert policy.check(undeclared).allowed is False
        mp.undo()


def test_policy_is_deterministic(monkeypatch):
    _deny(monkeypatch, network=False)
    cap = _cap("research", NETWORK_ONLY)
    assert policy.check(cap) == policy.check(cap)


def test_active_summary_reports_what_is_switched_off(monkeypatch):
    assert policy.active_summary() == "unrestricted"
    _deny(monkeypatch, network=False, subprocess=False)
    summary = policy.active_summary()
    assert "network" in summary and "subprocess" in summary
    assert "writes" not in summary


# --- Integration: the gate actually stops execution -------------------------


def _pin_tool(monkeypatch, name, handler, requirements):
    """
    Install a capability through the real derivation path: a tool in
    TOOLS (so the router parser accepts the name) whose REQUIREMENTS
    are pinned in the registry's cache. Going through TOOLS rather than
    REGISTERED matters -- the parser validates router output against
    the enabled tool set before dispatch is ever reached.
    """
    from forge.kernel import registry as capabilities
    from forge.tools import registry as tool_registry

    monkeypatch.setitem(tool_registry.TOOLS, name, handler)
    monkeypatch.setitem(capabilities._REQUIREMENTS, name, (requirements, True))


def test_orchestrator_refuses_a_denied_capability(monkeypatch):
    """
    The denial must reach the user as an explained refusal, not as a
    confusing downstream error from a tool that should never have run.
    """
    import forge.orchestrator as orch_mod

    ran = []

    def handler(_content):
        ran.append(True)
        return "should never run"

    _pin_tool(monkeypatch, "netthing", handler, NETWORK_ONLY)
    _deny(monkeypatch, network=False)
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "netthing", "content": "go"}),
    )

    result = Orchestrator().run("fetch something")

    assert ran == [], "a denied capability must not have its handler called"
    assert result.ok is False
    assert "network access" in (result.error or "")


def test_orchestrator_still_runs_an_allowed_capability(monkeypatch):
    """The gate must not become a blanket refusal once any flag is off."""
    import forge.orchestrator as orch_mod

    _pin_tool(monkeypatch, "localthing", lambda c: f"ran:{c}", LOCAL_READONLY)
    _deny(monkeypatch, network=False, writes=False, subprocess=False)
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "localthing", "content": "ok"}),
    )

    result = Orchestrator().run("do something local")
    assert result.ok is True
    assert result.output == "ran:ok"


def test_the_shipped_defaults_are_permissive(monkeypatch):
    """
    The claim the autouse fixture in conftest.py would otherwise hide:
    an untouched deployment must behave exactly as it did before the
    Policy Engine existed. Checked against forge.config with the
    variables absent, not against the pinned test values.
    """
    import importlib

    import forge.config as config_mod

    for var in (
        "POLICY_ALLOW_NETWORK",
        "POLICY_ALLOW_WORKSPACE_WRITES",
        "POLICY_ALLOW_SUBPROCESS",
    ):
        monkeypatch.delenv(var, raising=False)

    try:
        importlib.reload(config_mod)
        assert config_mod.POLICY_ALLOW_NETWORK is True
        assert config_mod.POLICY_ALLOW_WORKSPACE_WRITES is True
        assert config_mod.POLICY_ALLOW_SUBPROCESS is True
    finally:
        monkeypatch.undo()
        importlib.reload(config_mod)


# --- The router is not offered what the policy will refuse ------------------


def test_allowed_names_drops_denied_capabilities(monkeypatch):
    from forge.kernel import registry as capabilities

    _deny(monkeypatch, network=False)
    names = capabilities.allowed_names()

    assert "chat" in names
    assert "web_search" not in names
    assert "research" not in names


def test_allowed_names_is_everything_when_unrestricted():
    from forge.kernel import registry as capabilities
    from forge.tools import registry as tool_registry

    tool_registry.load_tools()
    assert capabilities.allowed_names() == capabilities.capability_names()


def test_the_router_prompt_never_offers_a_denied_capability(monkeypatch):
    """
    The point of the whole thing: a capability the policy refuses must
    not appear in the router prompt at all.

    Offering it costs a full routing call -- the dominant per-call cost
    on this box -- to reach a refusal that was knowable before the call
    was made, and it tells the model mid-conversation that a named tool
    does not work, which is what destabilises a 9B router.
    """
    from forge.router import build_router_prompt
    from forge.tools import registry as tool_registry

    def _noop(content: str) -> str:
        return content

    monkeypatch.setattr(
        tool_registry,
        "TOOLS",
        {"chat": _noop, "code": _noop, "web_search": _noop, "research": _noop},
    )

    offered = build_router_prompt("cherche des infos sur les lunettes AR")
    assert "web_search" in offered

    _deny(monkeypatch, network=False)
    withheld = build_router_prompt("cherche des infos sur les lunettes AR")
    assert "web_search" not in withheld
    assert "research" not in withheld
    assert "chat" in withheld


def test_a_fully_denied_tool_set_still_leaves_a_usable_prompt(monkeypatch):
    """
    The filter must degrade, never empty out. build_router_prompt falls
    back to a fixed pair when nothing is allowed, so a maximally
    restrictive policy yields a router that can still answer rather
    than one with no tools at all.
    """
    from forge.router import build_router_prompt
    from forge.tools import registry as tool_registry

    def _noop(content: str) -> str:
        return content

    monkeypatch.setattr(tool_registry, "TOOLS", {"shell": _noop})
    _deny(monkeypatch, network=False, writes=False, subprocess=False)

    prompt = build_router_prompt("fais quelque chose")
    assert "shell" not in prompt
    assert prompt.strip()
    assert "chat" in prompt


# --- Paths that reach a capability without dispatching ----------------------


def test_the_review_graph_skips_its_tests_when_subprocesses_are_denied(monkeypatch):
    """
    graphs/review reaches the `test` capability by importing the module,
    not by dispatching, so the orchestrator's gate never saw it. Under
    POLICY_ALLOW_SUBPROCESS=false it would have spawned pytest anyway.

    Checked against the capability it actually uses rather than review's
    own profile: review still runs (it declares subprocess=False and
    that stays true), and only the step that really spawns a process is
    withheld -- with a reason that reaches the review itself.
    """
    from forge.graphs import review as review_graph
    from forge.tools import registry as tool_registry
    from forge.types import AgentState

    spawned = []
    monkeypatch.setattr(
        review_graph.test_tool,
        "run",
        lambda content: spawned.append(content) or "all tests passed",
    )
    monkeypatch.setitem(tool_registry.TOOLS, "test", lambda c: c)

    state = AgentState(user_input="review it", max_steps=1)
    state.context["test_path"] = "tests/test_graph.py"

    review_graph._run_tests_node(state)
    assert spawned, "with subprocesses allowed the tests must actually run"

    spawned.clear()
    _deny(monkeypatch, subprocess=False)
    state.context["test_output"] = None
    review_graph._run_tests_node(state)

    assert spawned == [], "a denied subprocess must not be spawned"
    assert "skipped" in state.context["test_output"]
    assert "subprocesses" in state.context["test_output"]


def test_the_review_graph_runs_tests_when_test_is_not_a_capability(monkeypatch):
    """
    The gate subtracts from what is reachable; it must not add a new
    reason for something to stop working. This graph has always called
    tools/test.py directly, without `test` needing to be in
    ENABLED_TOOLS, so an unregistered `test` capability means no
    opinion -- not a denial.
    """
    from forge.graphs import review as review_graph
    from forge.tools import registry as tool_registry
    from forge.types import AgentState

    spawned = []
    monkeypatch.setattr(
        review_graph.test_tool,
        "run",
        lambda content: spawned.append(content) or "ok",
    )
    monkeypatch.setattr(tool_registry, "TOOLS", {"chat": lambda c: c})

    state = AgentState(user_input="review it", max_steps=1)
    state.context["test_path"] = "tests/test_graph.py"
    review_graph._run_tests_node(state)

    assert spawned, "no registered `test` capability means no opinion, not a denial"


def test_the_repl_command_names_what_is_blocked_and_why(capsys, monkeypatch):
    """
    A denied capability is not in the router prompt, so from inside the
    conversation it simply does not exist. Without somewhere to ask,
    the only symptom of a restrictive policy is Forge quietly not doing
    something -- which is indistinguishable from it being bad at its
    job.
    """
    from forge.main import _handle_capabilities
    from forge.tools import registry as tool_registry

    monkeypatch.setattr(
        tool_registry,
        "TOOLS",
        {"chat": lambda c: c, "web_search": lambda c: c},
    )
    _deny(monkeypatch, network=False)

    _handle_capabilities()
    out = capsys.readouterr().out

    assert "chat" in out
    assert "web_search" in out
    assert "network" in out
    assert "denying network" in out
