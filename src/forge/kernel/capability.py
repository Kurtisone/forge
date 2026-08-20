"""
The Capability interface -- what the Kernel asks for, never who answers.

A Capability is "something Forge can do", described independently of its
implementation. See ARCHITECTURE.md, Niveau 2: the Router asks for a
capability, the Registry lists candidates, and (later) the Scheduler
chooses one.

Today every capability is backed by exactly one provider, so this layer
buys nothing at dispatch time. That is deliberate -- it is phase 1 of
the three-phase rule: Primitive. The abstraction exists and is
inspectable; it is not yet load-bearing. Phase 2 (Observable) adds
measurement, phase 3 (Optimisable) adds the Scheduler that decides.

WHAT IS DELIBERATELY ABSENT: cost, latency and quality scores.
ARCHITECTURE.md lists them as candidate characteristics, and they
belong here eventually -- but nothing measures them yet. Hardcoding
plausible-looking numbers now would give a future Scheduler the
appearance of being informed while it decided on fiction, which is
precisely the 1 -> 3 jump the three-phase rule exists to prevent. They
arrive in phase 2, measured, or not at all.

`Requirements` below holds the other half instead: what is *statically*
true of a provider and knowable without running it. That half turns out
to be exactly what a Policy Engine consumes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Requirements:
    """
    Static, policy-relevant facts about a provider.

    Every field answers a question the Policy Engine will ask before
    allowing an execution, and every one is answerable by reading the
    provider's source -- no run, no measurement.

    - network            reaches the public Internet
    - llm                makes an LLM *generation* call of its own,
                         on top of the router's
    - mutates_workspace  can create or modify files under WORKSPACE_DIR
    - spawns_process     runs a subprocess

    Every field defaults to True. An undeclared provider must therefore
    look maximally demanding, so that a policy forbidding network access
    can never let one slip through by omission. Declaring is opt-in to
    *fewer* permissions -- the same fail-closed shape as ENABLED_TOOLS,
    which is opt-in to being reachable at all.
    """

    network: bool = True
    llm: bool = True
    mutates_workspace: bool = True
    spawns_process: bool = True

    def summary(self) -> str:
        """Human-readable one-liner, for `forge capabilities` and logs."""
        flags = [
            label
            for label, on in (
                ("network", self.network),
                ("llm", self.llm),
                ("writes", self.mutates_workspace),
                ("subprocess", self.spawns_process),
            )
            if on
        ]
        return ", ".join(flags) if flags else "local, read-only"


#: The opposite pole of the conservative default: a provider that needs
#: nothing. Spelled out once here so tool modules declare intent rather
#: than repeating four `False`s.
LOCAL_READONLY = Requirements(
    network=False,
    llm=False,
    mutates_workspace=False,
    spawns_process=False,
)


class Capability(Protocol):
    """
    Structural contract for anything the Registry can list.

    `execute` keeps the exact `str -> str` shape of a tool's `run()`, on
    purpose. When the orchestrator eventually dispatches through the
    Registry instead of get_tool(), its existing ToolResult construction
    and error handling apply unchanged -- the swap is a call-site edit,
    not a rewrite of the execution path.
    """

    name: str
    provider: str
    requirements: Requirements
    declared: bool

    def execute(self, content: str) -> str: ...


@dataclass(frozen=True)
class ToolCapability:
    """
    Adapter exposing an existing tool's `run(content) -> str` as a
    Capability, without touching the tool itself.

    This is the migration hinge from ARCHITECTURE.md: `Capability`
    wraps `Tool` rather than renaming it globally, so every existing
    test keeps passing and no dispatch path changes.

    `name` is the capability, `provider` the implementation. Today they
    are always equal -- one tool, one capability, one candidate. Keeping
    them as two fields rather than one is what makes the 1:1 visible as
    a current fact instead of baking it in as an assumption: the day a
    second provider answers for `search`, only the Registry changes.
    """

    name: str
    provider: str
    handler: Callable[[str], str]
    requirements: Requirements = Requirements()
    declared: bool = False

    def execute(self, content: str) -> str:
        return self.handler(content)
