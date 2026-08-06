"""
Capability Registry -- passive knowledge of "who can do what".

Per ARCHITECTURE.md the Registry lists candidates and *never chooses*.
That constraint is enforced here by omission: there is no resolve(),
no best(), no pick(). Choosing is the Cognitive Scheduler's job, and
the Scheduler does not exist yet. Adding a convenience "just give me
the one" helper now would quietly make this module the decision-maker
and leave nowhere for the Arbiter to slot in later.

It is a *view*, not a snapshot. candidates() derives tool-backed
capabilities from forge.tools.registry.TOOLS at call time, and the
handler itself is resolved at execution time. Neither the set of tools
nor the function behind one is cached here. That is deliberate:
anything that edits TOOLS -- load_tools(), a test pinning
ENABLED_TOOLS, a future hot-reload -- would otherwise leave dispatch
pointed at a tool set that no longer exists. A view cannot go stale,
so the failure mode is removed rather than guarded against.

The ENABLED_TOOLS gate therefore still applies, unchanged: a tool not
opted in never reaches TOOLS, so it is never a candidate here. This
layer adds no reachability of its own.

Requirements are read from an optional module-level REQUIREMENTS
constant on each tool. A tool that declares nothing is still listed,
with the conservative default (everything True) and marked undeclared
-- never dropped, never assumed harmless. Same no-silent-failure
posture as forge.tools.registry: a surprise shows up in the logs and
in `forge capabilities` rather than vanishing.
"""

import importlib
from collections.abc import Callable

from forge.errors import ToolNotFoundError
from forge.kernel.capability import Capability, Requirements, ToolCapability
from forge.logger import log
from forge.tools import registry as tool_registry

#: Explicitly registered providers, keyed by capability name. Empty in
#: normal operation today -- every capability is tool-backed. This is
#: where a provider that was never a Tool (a MemoryProvider, an OCR
#: engine, a second LLM) will register itself.
REGISTERED: dict[str, list[Capability]] = {}

#: Memoised per tool module. REQUIREMENTS is a constant of the module,
#: so reading it once is safe. Which tools *exist* is deliberately not
#: cached -- see the module docstring.
_REQUIREMENTS: dict[str, tuple[Requirements, bool]] = {}


def _requirements_for(tool_name: str) -> tuple[Requirements, bool]:
    """
    Read a tool module's declared REQUIREMENTS.

    Returns (requirements, declared). The module is fetched via
    importlib rather than passed in because forge.tools.registry stores
    only the bare run() handler, not the module it came from -- reading
    it back here keeps that module untouched. The import is a cache
    hit: load_tools() has already imported it.
    """
    if tool_name in _REQUIREMENTS:
        return _REQUIREMENTS[tool_name]

    try:
        mod = importlib.import_module(f"forge.tools.{tool_name}")
    except Exception as e:  # noqa: BLE001
        log.warning("cannot read requirements for tool %r: %s", tool_name, e)
        return Requirements(), False

    found = getattr(mod, "REQUIREMENTS", None)
    if not isinstance(found, Requirements):
        log.warning(
            "tool %r declares no REQUIREMENTS, assuming the most demanding "
            "profile (network, llm, writes, subprocess)",
            tool_name,
        )
        result = (Requirements(), False)
    else:
        result = (found, True)

    _REQUIREMENTS[tool_name] = result
    return result


def _late_bound(tool_name: str) -> Callable[[str], str]:
    """
    Resolve the handler at execution time, not when the capability is
    built. forge.tools.registry.TOOLS stays the single source of truth
    for which function answers for a tool.
    """

    def call(content: str) -> str:
        handler = tool_registry.get_tool(tool_name)
        if handler is None:
            raise ToolNotFoundError(f"tool {tool_name!r} is no longer registered")
        return handler(content)

    call.__name__ = f"late_bound_{tool_name}"
    return call


def _tool_capability(tool_name: str) -> ToolCapability:
    requirements, declared = _requirements_for(tool_name)
    return ToolCapability(
        name=tool_name,
        provider=tool_name,
        handler=_late_bound(tool_name),
        requirements=requirements,
        declared=declared,
    )


def register(capability: Capability) -> None:
    """
    Add an explicit, non-tool-backed candidate for a capability.

    Registration order is preserved, and explicit providers are listed
    before the tool-backed one, so a future MemoryProvider reads as a
    first-class candidate rather than an addendum to the tool that
    happened to exist first.
    """
    REGISTERED.setdefault(capability.name, []).append(capability)
    log.event(
        "capability.registered",
        name=capability.name,
        provider=capability.provider,
        requires=capability.requirements.summary(),
        declared=capability.declared,
    )


def candidates(name: str) -> list[Capability]:
    """Every provider able to answer for `name`. Empty list if none."""
    found: list[Capability] = list(REGISTERED.get(name, []))
    if name in tool_registry.TOOLS:
        found.append(_tool_capability(name))
    return found


def capability_names() -> list[str]:
    return sorted(set(REGISTERED) | set(tool_registry.TOOLS))


def undeclared() -> list[Capability]:
    """Providers running on the conservative default, for reporting."""
    return [
        cap
        for name in capability_names()
        for cap in candidates(name)
        if not cap.declared
    ]
