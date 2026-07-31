"""
Autonomous memory tool, dispatchable by the router (v3.7).

Same forge.rag backend as the REPL commands (!remember/!recall) and
the HTTP endpoints (/remember, /search) -- this is a third entry
point into the same storage, called in-process. The difference is
who initiates it: here it's the model itself, deciding from the
conversation that something is worth storing or worth looking up,
rather than a human typing an explicit command.

Interface (consistent with all Forge tools):
    run(content: str) -> str

content is a JSON instruction:
    {"action": "remember", "kind": "decision"|"todo", "content": "...", "project": "..."}
    {"action": "recall", "query": "...", "top_k": 5, "kind": "...", "project": "..."}

"project" is always optional. "top_k"/"kind"/"project" on recall are
also optional (top_k defaults to 5).

To activate this tool add it to ENABLED_TOOLS in .env.local:
    ENABLED_TOOLS=chat,code,memory
"""

import json

from forge import rag
from forge.logger import log


def _remember(instruction: dict) -> str:
    kind = instruction.get("kind", "").strip().lower()
    text = instruction.get("content", "").strip()
    project = instruction.get("project") or None

    if kind not in ("decision", "todo"):
        return "[error] 'remember' requires kind to be 'decision' or 'todo'"
    if not text:
        return "[error] 'remember' requires a non-empty 'content' field"

    conn = rag.get_connection()
    try:
        try:
            entry_id = rag.remember(conn, kind=kind, content=text, project=project)
        except rag.EmbeddingError as e:
            log.error("memory tool: remember failed: %s", e)
            return f"[error] remember failed: embedding server unreachable ({e})"
    finally:
        conn.close()

    log.event("memory.remember", entry_id=entry_id, kind=kind, project=project)
    return f"Remembered (#{entry_id})."


def _recall(instruction: dict) -> str:
    query = instruction.get("query", "").strip()
    if not query:
        return "[error] 'recall' requires a non-empty 'query' field"

    top_k = instruction.get("top_k", 5)
    kind = instruction.get("kind") or None
    project = instruction.get("project") or None

    conn = rag.get_connection()
    try:
        try:
            results = rag.search(
                conn, query=query, top_k=top_k, kind=kind, project=project
            )
        except rag.EmbeddingError as e:
            log.error("memory tool: recall failed: %s", e)
            return f"[error] recall failed: embedding server unreachable ({e})"
    finally:
        conn.close()

    log.event("memory.recall", query=query, hits=len(results))

    if not results:
        return "No matching memory found."

    lines = []
    for r in results:
        proj = f"/{r['project']}" if r["project"] else ""
        lines.append(f"- [{r['kind']}{proj}] {r['content']}")
    return "\n".join(lines)


def run(content: str) -> str:
    """
    Execute a memory operation described by a JSON instruction.

    Expected shapes:
        {"action": "remember", "kind": "decision", "content": "...", "project": "forge"}
        {"action": "recall", "query": "...", "top_k": 5}
    """
    try:
        instruction = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return f"[error] memory tool expects JSON, got: {content[:80]!r}"

    action = instruction.get("action", "").strip().lower()

    if action == "remember":
        return _remember(instruction)
    if action == "recall":
        return _recall(instruction)
    return f"[error] unknown action {action!r} (use: remember / recall)"
