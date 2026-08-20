"""
Policy Engine -- what is allowed to run, given the current context.

ARCHITECTURE.md places this as a transversal component rather than an
advanced extra, and this is its first, deliberately small form: a
deterministic deny gate over the static Requirements each capability
declares. No metrics, no learning, no scoring. Given the same config
it always returns the same verdict, and the verdict carries its own
reason -- the auditability requirement applies from the first version,
not from v2.

Scope, precisely:

- It only ever SUBTRACTS. A capability must already be reachable
  (in ENABLED_TOOLS, therefore in TOOLS, therefore a candidate) before
  the policy is even consulted. Setting a flag to true never grants
  anything; it only stops removing it.
- It denies by declared requirement, not by tool name. There is no
  list of forbidden tools to keep in sync -- a new tool that declares
  network=true is covered by POLICY_ALLOW_NETWORK the day it lands,
  and a tool that declares nothing is covered by all of them, since
  undeclared requirements default to the most demanding profile.
- It is not a sandbox. The real containment lives in the tools
  themselves (web_fetch's SSRF guard, files' workspace confinement,
  shell's allowlist). This gate is about context -- offline, metered,
  read-only machine -- not about defending against a hostile tool.

Everything defaults to allowed, so an untouched deployment behaves
exactly as it did before this module existed.
"""

from dataclasses import dataclass

from forge.config import (
    POLICY_ALLOW_NETWORK,
    POLICY_ALLOW_SUBPROCESS,
    POLICY_ALLOW_WORKSPACE_WRITES,
)
from forge.kernel.capability import Capability
from forge.logger import log


@dataclass(frozen=True)
class Verdict:
    """
    The outcome of a policy check, with its reason attached.

    `reason` is populated on denial and is meant to be shown, not just
    logged: a capability that silently does not run is indistinguishable
    from one that ran badly.
    """

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOWED = Verdict(allowed=True)


def check(capability: Capability) -> Verdict:
    """
    Decide whether `capability` may run under the active policy.

    Checked in declaration order so the reason is stable for a given
    capability rather than depending on which flag was flipped last.
    """
    requirements = capability.requirements

    denials = [
        (requirements.network and not POLICY_ALLOW_NETWORK, "network access"),
        (
            requirements.mutates_workspace and not POLICY_ALLOW_WORKSPACE_WRITES,
            "workspace writes",
        ),
        (requirements.spawns_process and not POLICY_ALLOW_SUBPROCESS, "subprocesses"),
    ]

    blocked = [what for denied, what in denials if denied]
    if not blocked:
        return ALLOWED

    reason = (
        f"{capability.name!r} requires {' and '.join(blocked)}, "
        "which the current policy does not allow"
    )
    log.event(
        "policy.denied",
        capability=capability.name,
        provider=capability.provider,
        blocked=blocked,
    )
    return Verdict(allowed=False, reason=reason)


def active_summary() -> str:
    """One-line description of the active policy, for `forge capabilities`."""
    off = [
        label
        for label, on in (
            ("network", POLICY_ALLOW_NETWORK),
            ("writes", POLICY_ALLOW_WORKSPACE_WRITES),
            ("subprocess", POLICY_ALLOW_SUBPROCESS),
        )
        if not on
    ]
    return "unrestricted" if not off else f"denying {', '.join(off)}"
