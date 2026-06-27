"""
Typed data structures that flow between the layers of Forge.

Nothing in the runtime should pass raw dicts across a module boundary.
This is what makes "LLM / tools / logs" a real separation instead of
a convention people forget after two commits.
"""

import time
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


@dataclass
class TraceStep:
    """
    One step in the execution of a single run().

    Mutable because it is filled in progressively as the step
    executes: decision is set by _route(), result and duration_ms
    are filled in by _dispatch().
    """
    step: int
    started_at: float = field(default_factory=time.monotonic)
    decision_tool: Optional[str] = None
    decision_content: Optional[str] = None
    router_raw: str = field(repr=False, default="")
    tool_ok: Optional[bool] = None
    tool_error: Optional[str] = None
    duration_ms: Optional[int] = None

    def finish(self, result: "ToolResult") -> None:
        self.tool_ok = result.ok
        self.tool_error = result.error
        self.duration_ms = int((time.monotonic() - self.started_at) * 1000)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "decision_tool": self.decision_tool,
            "decision_content": self.decision_content,
            "tool_ok": self.tool_ok,
            "tool_error": self.tool_error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentState:
    """
    The single object that travels through an entire run().

    Replaces the scattered local variables (seen_calls, step counter,
    partial results) that previously lived inside the run() loop.
    Carrying state explicitly makes each step inspectable and makes
    the execution trace a natural by-product, not an afterthought.
    """
    user_input: str
    max_steps: int
    history: list[dict] = field(default_factory=list)
    seen_calls: set = field(default_factory=set)
    steps_taken: int = 0
    trace: list[TraceStep] = field(default_factory=list)
    final_output: Optional[str] = None
    final_tool: Optional[str] = None
    ok: bool = True
    error: Optional[str] = None
    # Arbitrary key/value store for inter-node data passing in a graph.
    # Nodes can write results here (e.g. file content) and downstream
    # nodes can read them, without needing direct coupling.
    # Example: read_file_node writes context["file_content"] = "...",
    #          llm_review_node reads it.
    context: dict = field(default_factory=dict)

    def new_step(self) -> TraceStep:
        self.steps_taken += 1
        ts = TraceStep(step=self.steps_taken)
        self.trace.append(ts)
        return ts

    def current_step(self) -> Optional[TraceStep]:
        return self.trace[-1] if self.trace else None

    def to_result(self) -> "AgentResult":
        return AgentResult(
            output=self.final_output or "",
            tool=self.final_tool or "none",
            steps=self.steps_taken,
            ok=self.ok,
            error=self.error,
            trace=self.trace,
        )


@dataclass(frozen=True)
class AgentResult:
    """What run_agent() / Orchestrator.run() ultimately returns."""
    output: str
    tool: str
    steps: int
    ok: bool = True
    error: Optional[str] = None
    trace: list = field(default_factory=list, compare=False)

