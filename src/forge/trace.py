"""
Structured execution trace for Forge.

Every run produces a Trace — a timestamped, IDed record of every step
taken (router decision, tool dispatch, result, duration). Traces are
appended as JSONL to TRACE_FILE (one JSON object per line, default:
data/traces.jsonl), which makes them grep-able, cat-able, and readable
with `jq` without any tooling.

Two observability layers, two different purposes:
- SHOW_DEBUG=true: verbose real-time log for watching a run live
- trace (TRACE_ENABLED=true): durable per-run record for post-hoc
  analysis ("why did it pick that tool?", "how long did routing take?")

The trace is best-effort: a write failure logs a warning and is ignored.
It never affects the result of a run.
"""

import json
import time
import uuid
from pathlib import Path

from forge.config import TRACE_ENABLED, TRACE_FILE
from forge.logger import log


def _build_record(state) -> dict:
    """Build a serializable trace record from an AgentState."""
    steps = []
    for ts in state.trace:
        steps.append({
            "step": ts.step,
            "router_tool": ts.decision_tool,
            "router_content_preview": (ts.decision_content or "")[:120],
            "tool_ok": ts.tool_ok,
            "tool_error": ts.tool_error,
            "duration_ms": ts.duration_ms,
        })
    return {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user_input_preview": state.user_input[:120],
        "steps": steps,
        "final_tool": state.final_tool,
        "ok": state.ok,
        "error": state.error,
        "total_ms": sum(
            (s.get("duration_ms") or 0) for s in steps
        ),
    }


def save(state) -> None:
    """
    Append a trace record for this run. Called by the orchestrator at
    the end of every run(), regardless of success or failure.
    """
    if not TRACE_ENABLED:
        return
    try:
        record = _build_record(state)
        path = Path(TRACE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.event("trace.saved", run_id=record["run_id"], total_ms=record["total_ms"])
    except Exception as e:  # noqa: BLE001
        log.warning("failed to write trace: %s", e)


def read_last(n: int = 5) -> list[dict]:
    """Read the last n trace records from the JSONL file."""
    path = Path(TRACE_FILE)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        result = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(result) >= n:
                break
        return list(reversed(result))
    except OSError as e:
        log.warning("failed to read trace: %s", e)
        return []


def format_for_display(traces: list[dict]) -> str:
    """Human-readable summary for the !trace REPL command."""
    if not traces:
        return "(no traces yet)"
    lines = []
    for t in traces:
        ok_mark = "✓" if t.get("ok") else "✗"
        steps = t.get("steps", [])
        step_summary = " → ".join(
            s.get("router_tool") or "?" for s in steps
        )
        lines.append(
            f"[{ok_mark}] {t.get('timestamp')}  #{t.get('run_id')}  "
            f"{t.get('total_ms')}ms  {step_summary or 'no steps'}"
        )
        lines.append(f"    in:  {t.get('user_input_preview', '')!r}")
        if not t.get("ok") and t.get("error"):
            lines.append(f"    err: {t.get('error')}")
    return "\n".join(lines)
