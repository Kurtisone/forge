"""
Default Forge graph: router → dispatch → (fallback on error).

This is the graph equivalent of what the Orchestrator does today,
expressed as explicit nodes and conditional edges. It serves two
purposes:
1. A working example of the graph engine.
2. A drop-in that can replace Orchestrator.run() for multi-step flows
   once more nodes (files, shell) are activated.

Node order:
  [router] ──────────────────────────► [dispatch] ──► (terminal)
                                            │
                                    ok=False│
                                            ▼
                                       [fallback]  ──► (terminal)
"""

from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.router import build_router_prompt, parse_router_output
from forge.tools.registry import get_tool
from forge.types import AgentState


def _router_node(state: AgentState) -> AgentState:
    """Call the LLM router and record the decision on state."""
    try:
        prompt = build_router_prompt(state.user_input, history=state.history)
        raw = call_llm(prompt)
        decision = parse_router_output(raw)
        # Store decision for the dispatch node via a transient attribute.
        # We use state directly rather than a side-channel dict so the
        # trace captures it naturally.
        state._decision = decision  # type: ignore[attr-defined]
        state.final_tool = decision.tool
        log.event("graph.router", tool=decision.tool)
    except ProviderError as e:
        log.error("graph router: provider failure: %s", e)
        state.ok = False
        state.error = str(e)
        state.final_output = "The model backend is unavailable."
    return state


def _dispatch_node(state: AgentState) -> AgentState:
    """Run the tool chosen by the router node."""
    decision = getattr(state, "_decision", None)
    if decision is None:
        state.ok = False
        state.error = "dispatch: no router decision available"
        return state

    handler = get_tool(decision.tool)
    if handler is None:
        log.warning("graph dispatch: no handler for %r, using content", decision.tool)
        state.final_output = decision.content
        return state

    try:
        output = handler(decision.content)
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"tool {decision.tool!r} returned empty or non-str output")
        state.final_output = output
        log.event("graph.dispatch", tool=decision.tool, length=len(output))
    except Exception as e:  # noqa: BLE001
        log.error("graph dispatch: tool %r raised: %s", decision.tool, e)
        state.ok = False
        state.error = str(e)
        state.final_output = f"Tool error: {decision.tool}"
    return state


def _fallback_node(state: AgentState) -> AgentState:
    """
    Entered when dispatch failed. Returns a user-visible error message
    and resets ok=True so the caller gets a clean AgentResult rather
    than an exception.
    """
    log.warning("graph fallback: recovering from error: %s", state.error)
    state.final_output = (
        f"Something went wrong: {state.error or 'unknown error'}. "
        "Please try again."
    )
    state.ok = True   # error was handled; caller gets a result, not a crash
    state.error = None
    return state


def build() -> Graph:
    """Build and return the default router+dispatch+fallback graph."""
    from forge.config import MAX_STEPS
    g = Graph("default", max_steps=MAX_STEPS + 2)  # +2 for potential fallback

    g.add_node("router", _router_node)
    g.add_node("dispatch", _dispatch_node)
    g.add_node("fallback", _fallback_node)

    # router → dispatch (always, even if router marked ok=False — dispatch
    # will detect the missing decision and mark itself failed)
    g.add_edge("router", "dispatch")

    # dispatch → fallback only on failure
    g.add_edge("dispatch", "fallback", condition=lambda s: not s.ok)
    # dispatch → terminal on success (no edge needed, terminal by default
    # when no matching edge exists)

    return g
