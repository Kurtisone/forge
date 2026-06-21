"""
The orchestrator: the only place that knows how to go from user input
to a final answer. It calls the LLM (router), dispatches to a tool,
and returns a typed AgentResult.

This replaces agent.py. Three rules enforced here, by construction,
not by convention:

1. LLM / tools / logs stay separate: this module never calls
   requests.post() or print() itself -- it only calls call_llm(),
   tool handlers, and log.*().
2. No loop is possible: every run is bounded by MAX_STEPS, and seen
   (tool, content) pairs are tracked so a router that asks for the
   same step twice is stopped instead of spinning.
3. Every failure is a typed AgentResult, never a bare exception
   leaking to main.py or a silently-swallowed empty string.
4. Memory (conversation history) is best-effort: a read/write
   failure is logged and ignored, never allowed to break a turn.
"""

from forge import memory
from forge.config import MAX_STEPS, MEMORY_ENABLED
from forge.errors import LoopGuardError, ProviderError
from forge.llm import call_llm
from forge.logger import log
from forge.router import build_router_prompt, parse_router_output
from forge.tools.registry import get_tool, load_tools
from forge.types import AgentResult, ToolResult

load_tools()


class Orchestrator:
    def __init__(self, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps

    def run(self, user_input: str) -> AgentResult:
        seen_calls = set()

        for step in range(1, self.max_steps + 1):
            try:
                decision = self._route(user_input)
            except ProviderError as e:
                log.error("provider failure: %s", e)
                return AgentResult(
                    output="The model backend is unavailable.",
                    tool="none",
                    steps=step,
                    ok=False,
                    error=str(e),
                )

            call_signature = (decision.tool, decision.content)
            if call_signature in seen_calls:
                err = LoopGuardError(
                    f"repeated call to tool={decision.tool!r} "
                    f"with identical content detected, stopping"
                )
                log.error(str(err))
                return AgentResult(
                    output="Stopped: the router tried to repeat the same step.",
                    tool=decision.tool,
                    steps=step,
                    ok=False,
                    error=str(err),
                )
            seen_calls.add(call_signature)

            result = self._dispatch(decision.tool, decision.content)

            if MEMORY_ENABLED:
                self._remember(user_input, result.output)

            # Single-shot today: the first successful dispatch is the
            # final answer. When the roadmap's multi-step tools land,
            # this is the one line that changes (decide whether to
            # feed result.output back into the next step) -- the
            # guard above already protects that future loop.
            return AgentResult(
                output=result.output,
                tool=result.tool,
                steps=step,
                ok=result.ok,
                error=result.error,
            )

        # Unreachable while MAX_STEPS >= 1, kept for safety.
        raise LoopGuardError("max_steps exhausted without a result")

    def _remember(self, user_input: str, output: str) -> None:
        # Memory is a convenience layer: it must never fail a turn.
        try:
            memory.add_exchange(user_input, str(output))
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist memory: %s", e)

    def _route(self, user_input: str):
        history = self._recall()
        prompt = build_router_prompt(user_input, history=history)
        log.event("router.prompt", chars=len(prompt))

        raw = call_llm(prompt)
        log.event("router.raw_output", raw=raw)

        decision = parse_router_output(raw)
        log.event("router.decision", tool=decision.tool, content=decision.content)
        return decision

    def _recall(self) -> list[dict]:
        if not MEMORY_ENABLED:
            return []
        try:
            return memory.get_history()
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load memory: %s", e)
            return []

    def _dispatch(self, tool: str, content: str):
        handler = get_tool(tool)
        if handler is None:
            log.warning("no handler registered for tool %r, returning content as-is", tool)
            return ToolResult(tool=tool, output=content, ok=True)

        log.event("tool.dispatch", tool=tool)
        try:
            output = handler(content)
        except Exception as e:  # noqa: BLE001 - a tool must never crash the runtime
            log.error("tool %r raised: %s", tool, e)
            return ToolResult(
                tool=tool, output=f"Tool error: {tool}", ok=False, error=str(e)
            )

        log.event("tool.result", tool=tool, length=len(str(output)))
        return ToolResult(tool=tool, output=output, ok=True)


def run_agent(user_input: str) -> str:
    """
    Backward-compatible function, same signature as the old agent.py.
    main.py (and anything else) can keep calling run_agent() unchanged.
    """
    result = Orchestrator().run(user_input)
    return result.output
