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
7. No tool escalation after external data: once a step has pulled in
   content Forge doesn't control, no later step of the same run may
   dispatch a mutating tool. See _EXTERNAL_INGEST_TOOLS below.
"""

import json

from forge import memory, metrics, subtrace, trace
from forge.config import (
    ALLOW_MUTATION_AFTER_EXTERNAL_DATA,
    MAX_STEPS,
    MEMORY_ENABLED,
)
from forge.errors import LoopGuardError, ProviderError, ToolExecutionError
from forge.llm import call_llm
from forge.logger import log
from forge.router import build_router_prompt, parse_router_output
from forge.tools.registry import get_tool, load_tools
from forge.types import AgentResult, AgentState, ToolResult

load_tools()

# --- Escalation guard (audit E-2) -------------------------------------
#
# Tools whose output is content Forge does not control: a web page, a
# search result page, a system log. From the second step onward that
# text sits in the prompt that picks the next tool (see step_context
# in run() below and router/prompt.py), so it is in a position to
# suggest one.
#
# files:read is deliberately NOT here. It reads inside WORKSPACE_DIR,
# and the read-then-write chain is a designed flow (v3.9: "remplace X
# par Y dans hello.go" reads the real file, then writes it back with
# the change applied). Taking files:read as tainting would break the
# one legitimate multi-step flow this project actually uses, to
# defend against a file the user's own workspace put there -- the
# wrong trade. Nothing stops a hostile page's content being written
# to the workspace in one turn and read back in another; that is a
# real limit of a per-run taint and it is stated in SECURITY.md
# rather than papered over here.
_EXTERNAL_INGEST_TOOLS = frozenset({"web_fetch", "web_search", "research", "sysadmin"})

# Tools that change something outside the run. "code" is not one: it
# returns source as text, it doesn't execute or persist anything.
_MUTATING_TOOLS = frozenset({"shell", "test"})

# files is both, depending on its action -- resolved by _is_mutating().
_FILES_READONLY_ACTIONS = frozenset({"read", "list"})

if ALLOW_MUTATION_AFTER_EXTERNAL_DATA:
    # Logged once at import, same reasoning as shell.py's allowlist
    # tripwire: a configuration fact belongs in the startup log, not
    # repeated into the output of every run where people learn to
    # scroll past it.
    log.warning(
        "orchestrator: ALLOW_MUTATION_AFTER_EXTERNAL_DATA is set -- a run "
        "that has fetched a web page or read system logs may go on to "
        "write files or run commands in a later step. The content of "
        "that page is part of the prompt choosing that step."
    )


def _is_mutating(tool: str, content: str) -> bool:
    """
    Does dispatching this call change anything outside the run?

    Fails closed for files: anything that isn't demonstrably a read or
    a list counts as a write. A malformed payload is exactly the case
    where guessing "probably harmless" is worst -- files.py's own
    parser is more forgiving than this check, and the gap between the
    two is where an escalation would live.
    """
    if tool in _MUTATING_TOOLS:
        return True
    if tool != "files":
        return False

    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return True
    if not isinstance(payload, dict):
        return True
    action = str(payload.get("action", "")).strip().lower()
    return action not in _FILES_READONLY_ACTIONS


class Orchestrator:
    def __init__(self, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps

    def run(self, user_input: str) -> AgentResult:
        # Reset before anything else: _recall() below can trigger
        # compaction, which calls the LLM, and that call belongs to
        # this run's bill.
        metrics.start_run()
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
                ts.abandon(f"provider failure: {e}")
                state.ok = False
                state.error = str(e)
                state.final_output = "The model backend is unavailable."
                state.final_tool = "none"
                return self._finish(state, remember=False)

            ts.decision_tool = decision.tool
            ts.decision_content = decision.content
            ts.router_raw = decision.raw

            # --- Loop guard ----------------------------------------------
            call_signature = (decision.tool, decision.content)
            if call_signature in state.seen_calls:
                if decision.tool == "web_search":
                    # A repeated call on the second step is a known,
                    # non-fatal failure mode with small local models: the
                    # steering hint in router/prompt.py asks them to
                    # switch to chat or web_fetch on the second step, but
                    # in practice they sometimes repeat the same tool
                    # call instead -- confirmed live with two different
                    # hint phrasings (prose-only, then an explicit worked
                    # JSON example) and with prompt caching disabled to
                    # rule out a cache-reuse bug, so this isn't a
                    # prompt-wording or infra problem, it's a genuine
                    # self-correction limit of this model class. state
                    # already holds the previous (successful) result at
                    # this point -- degrade to that instead of surfacing
                    # an internal loop-guard message. Every other tool
                    # still hard-fails below: a repeat there is a genuine
                    # signal worth surfacing, not something to paper
                    # over.
                    #
                    # memory used to need this same fallback, for the
                    # same underlying failure: routing memory:recall
                    # with "done": false and asking the router to
                    # phrase a real answer on the next step. That
                    # chaining is what looped, not memory itself -- see
                    # graphs/recall.py's docstring. "recall" is now one
                    # deterministic call that never re-enters this loop,
                    # so the tool that used to trip this branch can no
                    # longer reach it.
                    log.warning(
                        "%s tool repeated identical call (%r) -- "
                        "falling back to its previous result instead of "
                        "a loop-guard error",
                        decision.tool,
                        decision.content,
                    )
                    ts.abandon(
                        "loop guard: repeated call, fell back to previous result"
                    )
                    return self._finish(state, remember=True)

                err = LoopGuardError(
                    f"repeated call to tool={decision.tool!r} with identical content"
                )
                log.error(str(err))
                ts.abandon(str(err))
                state.ok = False
                state.error = str(err)
                state.final_output = (
                    "Stopped: the router tried to repeat the same step."
                )
                state.final_tool = decision.tool
                return self._finish(state, remember=False)
            state.seen_calls.add(call_signature)

            # --- Escalation guard ----------------------------------------
            # Checked before dispatch, never after: the point is that
            # the tool must not run, not that its effect gets reported.
            if (
                state.external_data_seen
                and not ALLOW_MUTATION_AFTER_EXTERNAL_DATA
                and _is_mutating(decision.tool, decision.content)
            ):
                note = (
                    f"escalation guard: tool={decision.tool!r} would mutate "
                    f"after {state.external_data_source!r} pulled in external "
                    "data earlier in this run"
                )
                log.error(note)
                ts.abandon(note)
                state.ok = False
                state.error = note
                state.final_output = (
                    "Stopped: this run already read outside data "
                    f"(via {state.external_data_source}), so it can no longer "
                    "write files or run commands. If that was what you "
                    "wanted, ask again as a separate request."
                )
                state.final_tool = decision.tool
                return self._finish(state, remember=False)

            # --- Dispatch ------------------------------------------------
            result = self._dispatch(decision.tool, decision.content)
            ts.finish(result)

            if decision.tool in _EXTERNAL_INGEST_TOOLS:
                # Marked on dispatch, not on success: a tool that
                # failed mid-way may still have put part of what it
                # read into the trace and the logs, and "it errored so
                # nothing came in" is an assumption about code this
                # module deliberately doesn't look inside.
                state.external_data_seen = True
                state.external_data_source = decision.tool

            state.final_output = result.output
            state.final_tool = result.tool
            state.ok = result.ok
            state.error = result.error

            # --- Continue or stop -----------------------------------------
            # decision.done defaults to True, so every extraction path that
            # predates this field (plain JSON, XML, markdown fence, plain
            # text fallback) still returns after exactly one step, exactly
            # like before. A tool failure also always stops the run — a
            # failed step is never a safe base to route again from. Only
            # an explicit "done": false, with steps still available and a
            # successful result, keeps the loop going.
            if not result.ok or decision.done or state.steps_taken >= self.max_steps:
                return self._finish(
                    state, remember=result.ok and not decision.is_fallback
                )

            # This step isn't final -- feed its tool result to the next
            # routing decision via step_context, NOT history. history
            # must stay an exact mirror of what's persisted to
            # memory.json (see _finish below), so that the router
            # prompt's history block is byte-identical between the last
            # call of one turn and the first call of the next, and
            # llama-server can reuse its KV cache for that whole prefix
            # instead of invalidating it every turn.
            state.step_context = state.step_context + [
                {"role": "assistant", "content": f"[{result.tool}] {result.output}"}
            ]

        raise LoopGuardError("max_steps exhausted without a result")

    # ------------------------------------------------------------------

    def _finish(self, state: AgentState, remember: bool) -> AgentResult:
        """
        Single exit point for every run(): saves the trace and, at
        most once per run, persists the turn to memory.json.

        Persisting here instead of once per step (the old behavior)
        fixes two things at once: memory.json no longer accumulates a
        duplicate, raw intermediate tool-result exchange alongside the
        real final answer for every multi-step run, and `history` (see
        _recall/_route) stays a stable, cacheable prefix across turns
        instead of drifting every time a run takes more than one step.
        """
        trace.save(state)
        if MEMORY_ENABLED and remember:
            self._remember(state.user_input, state.final_output or "")
        return state.to_result()

    def _route(self, state: AgentState):
        prompt = build_router_prompt(
            state.user_input, history=state.history, step_context=state.step_context
        )
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
            subtrace.clear()  # discard any stale publish, same as every other exit path
            return ToolResult(tool=tool, output=content, ok=True)

        log.event("tool.dispatch", tool=tool)
        subtrace.clear()  # start every dispatch on a clean slate -- see subtrace.clear()
        try:
            output = handler(content)
            output = self._validate_tool_output(tool, output)
        except ToolExecutionError as e:
            log.error("tool %r violated its contract: %s", tool, e)
            subtrace.pop()  # discard: a failed call's partial steps aren't useful
            return ToolResult(
                tool=tool, output=f"Tool error: {tool}", ok=False, error=str(e)
            )
        except Exception as e:  # noqa: BLE001
            log.error("tool %r raised: %s", tool, e)
            subtrace.pop()
            return ToolResult(
                tool=tool, output=f"Tool error: {tool}", ok=False, error=str(e)
            )

        sub_steps = subtrace.pop()
        log.event("tool.result", tool=tool, length=len(output))
        return ToolResult(tool=tool, output=output, ok=True, sub_steps=sub_steps)

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
        # Content used to be hard-truncated to _MAX_MEMORY_CONTENT chars
        # here (pre-v3.9), to keep the router's own prompt from
        # ballooning on large pastes/tool output. That's now the job of
        # compaction.py (message-count threshold + a real summary,
        # instead of a blind per-message cut) and MEMORY_MAX_HISTORY
        # (message-count cap) -- both introduced in v3.9. Truncating
        # here as well silently corrupted what's persisted, which the
        # v3.9 web UI now renders directly (GET /history) instead of
        # only feeding it back into the router's own prompt: a 300-char
        # cap on a tool result like a file read showed up as a broken
        # answer on screen, not just a shorter prompt.
        try:
            memory.add_exchange(user_input, output)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist memory: %s", e)


def run_agent(user_input: str) -> str:
    """Backward-compatible alias."""
    return Orchestrator().run(user_input).output
