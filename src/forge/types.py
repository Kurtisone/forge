"""
Typed data structures that flow between the layers of Forge.

Nothing in the runtime should pass raw dicts across a module boundary.
This is what makes "LLM / tools / logs" a real separation instead of
a convention people forget after two commits.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RouterDecision:
    """What the router LLM decided to do, already validated."""
    tool: str
    content: str
    raw: str = field(repr=False, default="")


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running a tool."""
    tool: str
    output: str
    ok: bool = True
    error: Optional[str] = None


@dataclass(frozen=True)
class AgentResult:
    """What run_agent() / Orchestrator.run() ultimately returns."""
    output: str
    tool: str
    steps: int
    ok: bool = True
    error: Optional[str] = None
