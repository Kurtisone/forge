"""
Tests for forge.kernel: the Capability interface, the ToolCapability
adapter, and the passive Registry.

Three properties matter more than the rest and are asserted directly
rather than left to documentation:

1. Fail-closed. A provider that declares no REQUIREMENTS is listed
   with every requirement on and flagged undeclared -- never dropped,
   never assumed harmless. Omission must not pass a policy.
2. The Registry never chooses. No resolve()/best()/pick(). Choosing
   belongs to the Cognitive Scheduler, which does not exist yet.
3. The Registry cannot go stale. It is a view over TOOLS, not a
   snapshot of it, so nothing it returns can outlive a reload.
"""

import dataclasses
import importlib
import sys

import pytest

import forge.tools as tools_pkg
from forge.errors import ToolNotFoundError
from forge.kernel import registry as capabilities
from forge.kernel.capability import LOCAL_READONLY, Requirements, ToolCapability
from forge.tools import registry as tool_registry

# --- Requirements -----------------------------------------------------------


def test_requirements_default_to_the_most_demanding_profile():
    """
    Omission must never look harmless: an undeclared provider has to
    fail a restrictive policy, not slip past it.
    """
    r = Requirements()
    assert r.network is True
    assert r.llm is True
    assert r.mutates_workspace is True
    assert r.spawns_process is True


def test_local_readonly_is_the_opposite_pole():
    assert LOCAL_READONLY == Requirements(
        network=False, llm=False, mutates_workspace=False, spawns_process=False
    )


def test_summary_lists_only_what_is_required():
    assert LOCAL_READONLY.summary() == "local, read-only"
    assert Requirements().summary() == "network, llm, writes, subprocess"
    assert (
        Requirements(
            network=True, llm=False, mutates_workspace=False, spawns_process=False
        ).summary()
        == "network"
    )


def test_requirements_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Requirements().network = False


# --- ToolCapability ---------------------------------------------------------


def test_tool_capability_preserves_the_str_to_str_contract():
    """
    execute() must keep run()'s exact shape, so the orchestrator's
    existing ToolResult construction and error handling still apply
    now that dispatch goes through the registry.
    """
    cap = ToolCapability(
        name="upper", provider="upper", handler=lambda c: c.upper(), declared=True
    )
    assert cap.execute("hi") == "HI"


def test_tool_capability_does_not_swallow_handler_errors():
    """
    The adapter must stay transparent: error handling lives in the
    orchestrator, and wrapping must not become a second place where
    failures are caught and reshaped.
    """

    def boom(_content):
        raise ValueError("handler failed")

    cap = ToolCapability(name="boom", provider="boom", handler=boom)
    with pytest.raises(ValueError, match="handler failed"):
        cap.execute("anything")


def test_capability_and_provider_are_separate_fields():
    cap = ToolCapability(name="search", provider="web_search", handler=lambda c: c)
    assert cap.name != cap.provider


# --- Registry: derived live from the tool registry --------------------------


def test_every_enabled_tool_is_a_capability():
    tool_registry.load_tools()
    assert capabilities.capability_names() == tool_registry.available_tools()


def test_a_tool_added_to_TOOLS_is_immediately_a_candidate(monkeypatch):
    """
    The registry is a view, not a snapshot. Something that edits TOOLS
    in place -- a test, a future hot-reload -- must be visible without
    any rebuild step, because a rebuild step is a thing someone can
    forget to call.
    """
    monkeypatch.setitem(tool_registry.TOOLS, "live_view_case", lambda c: "hi")
    found = capabilities.candidates("live_view_case")
    assert len(found) == 1
    assert found[0].execute("x") == "hi"


def test_the_handler_is_resolved_at_execution_time(monkeypatch):
    """
    Late binding: TOOLS stays the single source of truth for which
    function answers. A capability obtained before a swap must run the
    function that is current when execute() is called, not the one
    that was there when it was handed out.
    """
    monkeypatch.setitem(tool_registry.TOOLS, "late_bind_case", lambda c: "first")
    cap = capabilities.candidates("late_bind_case")[0]

    monkeypatch.setitem(tool_registry.TOOLS, "late_bind_case", lambda c: "second")
    assert cap.execute("x") == "second"


def test_executing_a_removed_tool_raises_rather_than_calling_a_ghost(monkeypatch):
    monkeypatch.setitem(tool_registry.TOOLS, "vanishing_case", lambda c: "here")
    cap = capabilities.candidates("vanishing_case")[0]

    monkeypatch.delitem(tool_registry.TOOLS, "vanishing_case")
    with pytest.raises(ToolNotFoundError, match="no longer registered"):
        cap.execute("x")


def test_registry_inherits_the_enabled_tools_gate(tmp_path, monkeypatch):
    """
    The kernel layer must add no reachability. A tool excluded by
    ENABLED_TOOLS never reaches TOOLS, so it must never become a
    Capability either.
    """
    (tmp_path / "tool_kernel_gate_case.py").write_text(
        "def run(content):\n    return 'unreachable'\n"
    )
    monkeypatch.setattr(tools_pkg, "__path__", [str(tmp_path)])
    monkeypatch.setattr(tool_registry, "ENABLED_TOOLS", set())
    sys.modules.pop("forge.tools.tool_kernel_gate_case", None)

    try:
        tool_registry.load_tools()
        assert capabilities.candidates("tool_kernel_gate_case") == []
    finally:
        sys.modules.pop("forge.tools.tool_kernel_gate_case", None)
        monkeypatch.undo()
        tool_registry.load_tools()


def test_undeclared_tool_gets_the_conservative_profile(tmp_path, monkeypatch):
    """
    The fail-closed guarantee end to end: a tool with a valid run() but
    no REQUIREMENTS is still listed (never dropped) and is marked
    undeclared with every requirement on.
    """
    (tmp_path / "tool_kernel_undeclared_case.py").write_text(
        "def run(content):\n    return content\n"
    )
    monkeypatch.setattr(tools_pkg, "__path__", [str(tmp_path)])
    monkeypatch.setattr(tool_registry, "ENABLED_TOOLS", {"tool_kernel_undeclared_case"})
    sys.modules.pop("forge.tools.tool_kernel_undeclared_case", None)
    capabilities._REQUIREMENTS.pop("tool_kernel_undeclared_case", None)

    try:
        tool_registry.load_tools()

        found = capabilities.candidates("tool_kernel_undeclared_case")
        assert len(found) == 1
        assert found[0].declared is False
        assert found[0].requirements == Requirements()
        assert [c.provider for c in capabilities.undeclared()] == [
            "tool_kernel_undeclared_case"
        ]
    finally:
        sys.modules.pop("forge.tools.tool_kernel_undeclared_case", None)
        capabilities._REQUIREMENTS.pop("tool_kernel_undeclared_case", None)
        monkeypatch.undo()
        tool_registry.load_tools()


def test_declared_requirements_are_read_from_the_tool_module():
    tool_registry.load_tools()

    chat = capabilities.candidates("chat")
    assert len(chat) == 1
    assert chat[0].declared is True
    assert chat[0].requirements == LOCAL_READONLY


def test_candidates_is_empty_for_an_unknown_capability():
    tool_registry.load_tools()
    assert capabilities.candidates("does_not_exist") == []


# --- Registry: explicit, non-tool-backed providers --------------------------


def test_an_explicit_provider_can_answer_alongside_a_tool(monkeypatch):
    """
    The 1:1 tool-to-capability mapping is a fact about today, not an
    assumption baked into the structure. This is the shape that will
    make the Cognitive Scheduler necessary.
    """
    monkeypatch.setitem(tool_registry.TOOLS, "search_case", lambda c: "from tool")
    monkeypatch.setitem(
        capabilities.REGISTERED,
        "search_case",
        [
            ToolCapability(
                name="search_case",
                provider="some_future_provider",
                handler=lambda c: "from provider",
                declared=True,
            )
        ],
    )

    found = capabilities.candidates("search_case")
    assert [c.provider for c in found] == ["some_future_provider", "search_case"]


def test_candidates_returns_a_copy(monkeypatch):
    """
    Callers must not be able to mutate the registry by editing a list
    they were handed -- it stays the only writer of its own state.
    """
    monkeypatch.setitem(tool_registry.TOOLS, "copy_case", lambda c: c)

    got = capabilities.candidates("copy_case")
    got.append("not a capability")
    assert len(capabilities.candidates("copy_case")) == 1


def test_registry_exposes_no_way_to_choose():
    """
    ARCHITECTURE.md: the Registry lists candidates and never chooses.
    Choosing belongs to the Cognitive Scheduler, which does not exist
    yet -- so any helper returning a single winner would quietly make
    this module the decision-maker and leave the Arbiter nowhere to
    slot in. Asserted, not just documented.
    """
    for forbidden in ("resolve", "best", "pick", "select", "choose"):
        assert not hasattr(capabilities, forbidden)


def test_every_shipped_tool_declares_its_requirements():
    """
    A forcing function, not a style check.

    An undeclared tool still works -- it is registered with the most
    demanding profile and flagged. But that profile denies it under any
    policy restriction, so a tool that simply forgot to declare would
    look broken in a degraded context for a reason nobody would think
    to look for. Failing here, at the moment the tool is added, is the
    cheap version of that discovery.

    Scans the package rather than a list, so a new tool is covered the
    day its file lands. The fix is one REQUIREMENTS constant in the
    module the failure names.
    """
    import pkgutil

    import forge.tools as tools_pkg
    from forge.kernel.capability import Requirements

    missing = []
    for module in pkgutil.iter_modules(tools_pkg.__path__):
        if module.name == "registry":
            continue
        mod = importlib.import_module(f"forge.tools.{module.name}")
        if not hasattr(mod, "run"):
            continue
        if not isinstance(getattr(mod, "REQUIREMENTS", None), Requirements):
            missing.append(module.name)

    assert not missing, (
        f"tools without a REQUIREMENTS declaration: {sorted(missing)} -- "
        "add one to each module (see forge/kernel/capability.py)"
    )
