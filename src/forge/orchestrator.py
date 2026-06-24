"""
The orchestrator: the only place that knows how to go from user input
to a final answer.

Rules enforced here, by construction:

1. LLM / tools / logs stay separate: this module never calls
   requests.post() or print() — only call_llm(), tool handlers,
   log.*(), and trace.save().
2. No loop is possible: every run is bounded by MAX_STEPS, and
   AgentState.seen_calls prevents the same (tool, content) pair
   from being dispatched twice in a single run.
3. Every failure is a typed AgentResult, never a bare exception
   leaking to main.py or a silently-swallowed empty string.
4. Memory is best-effort: a read/write failure is logged and ignored.
5. The ToolResult contract is enforced: a tool returning anything
   other than a non-empty str is a ToolExecutionError.
6. Execution is traceable: AgentState accumulates a TraceStep per
   step and trace.save() writes it to disk when TRACE_ENABLED=true.
"""

from forge import memory
from forge import trace
from forge.config import MAX_STEPS, MEMORY_ENABLED
from forge.errors import LoopGuardError, ProviderError, ToolExecutionError
from forge.llm import call_llm
from forge.logger import log
from forge.router import build_router_prompt, parse_router_output
from forge.tools.registry import get_tool, load_tools
from forge.types import AgentResult, AgentState, ToolResult

load_tools()

_MAX_MEMORY_CONTENT = 300


class Orchestrator:
    def __init__(self, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps

    def run(self, user_input: str) -> AgentResult:
        state = AgentState(
            user_input=user_input,
            max_steps=self.max_steps,
            history=self._recall(),
        )

        for _ in range(self.max_steps):
            ts = state.new_step()

            # --- Route ---------------------------------------------------
            try:
                decision = self._route(state)
            except ProviderError as e:
                log.error("provider failure: %s", e)
                state.ok = False
                state.error = str(e)
                state.final_output = "The model backend is unavailable."
                state.final_tool = "none"
                trace.save(state)
                return state.to_result()

            ts.decision_tool = decision.tool
            ts.decision_content = decision.content
            ts.router_raw = decision.raw

            # --- Loop guard ----------------------------------------------
            call_signature = (decision.tool, decision.content)
            if call_signature in state.seen_calls:
                err = LoopGuardError(
                    f"repeated call to tool={decision.tool!r} with identical content"
                )
                log.error(str(err))
                state.ok = False
                state.error = str(err)
                state.final_output = "Stopped: the router tried to repeat the same step."
                state.final_tool = decision.tool
                trace.save(state)
                return state.to_result()
            state.seen_calls.add(call_signature)

            # --- Dispatch ------------------------------------------------
            result = self._dispatch(decision.tool, decision.content)
            ts.finish(result)

            if MEMORY_ENABLED and result.ok:
                self._remember(user_input, result.output)

            state.final_output = result.output
            state.final_tool = result.tool
            state.ok = result.ok
            state.error = result.error
            trace.save(state)
            return state.to_result()

        raise LoopGuardError("max_steps exhausted without a result")

    # ------------------------------------------------------------------

    def _route(self, state: AgentState):
        prompt = build_router_prompt(state.user_input, history=state.history)
        log.event("router.prompt", chars=len(prompt))
        raw = call_llm(prompt)
        log.event("router.raw_output", raw=raw)
        decision = parse_router_output(raw)
        log.event("router.decision", tool=decision.tool, content=decision.content)
        return decision

    def _dispatch(self, tool: str, content: str) -> ToolResult:
        handler = get_tool(tool)
        if handler is None:
            log.warning("no handler for tool %r, returning content as-is", tool)
            return ToolResult(tool=tool, output=content, ok=True)

        log.event("tool.dispatch", tool=tool)
        try:
            output = handler(content)
            output = self._validate_tool_output(tool, output)
        except ToolExecutionError as e:
            log.error("tool %r violated its contract: %s", tool, e)
            return ToolResult(tool=tool, output=f"Tool error: {tool}", ok=False, error=str(e))
        except Exception as e:  # noqa: BLE001
            log.error("tool %r raised: %s", tool, e)
            return ToolResult(tool=tool, output=f"Tool error: {tool}", ok=False, error=str(e))

        log.event("tool.result", tool=tool, length=len(output))
        return ToolResult(tool=tool, output=output, ok=True)

    def _validate_tool_output(self, tool: str, output) -> str:
        if not isinstance(output, str):
            raise ToolExecutionError(
                f"tool {tool!r} must return str, got {type(output).__name__}"
            )
        if not output.strip():
            raise ToolExecutionError(f"tool {tool!r} returned empty output")
        return output

    def _recall(self) -> list[dict]:
        if not MEMORY_ENABLED:
            return []
        try:
            return memory.get_history()
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load memory: %s", e)
            return []

    def _remember(self, user_input: str, output: str) -> None:
        try:
            memory.add_exchange(
                user_input[:_MAX_MEMORY_CONTENT],
                output[:_MAX_MEMORY_CONTENT],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist memory: %s", e)


def run_agent(user_input: str) -> str:
    """Backward-compatible alias."""
    return Orchestrator().run(user_input).output
