"""
Forge REPL.

Single-line usage: just type and press Enter as usual.

Multi-line code paste — two ways:
  A) Paste question + code together (recommended):
       Forge > Optimise ce Containerfile
       FROM python:3.12          ← these lines arrive in one paste
       WORKDIR /app              ← select() detects it automatically
       ...
     No fences needed when pasting in one go.

  B) Manual fence mode — type ``` on an empty line to start,
     then ``` again on an empty line to submit:
       Forge > ```
       ... FROM python:3.12
       ... WORKDIR /app
       ... ```          ← closes the block and submits
     Use this only for code. Plain questions don't need fences.
     Type 'cancel' on an empty ... line to abort without submitting.

Special commands (prefix with !):
  !clear                              wipe conversation history so the next turn starts fresh
  !compact                            force a compaction pass now (pinned messages excluded, v3.9)
  !trace                              show the last 5 execution traces
  !remember <decision|todo|fact> <project|-> <content>
                                       store a decision/todo/fact in vector memory (v3.7)
  !recall <query>                     semantic search over remembered entries
  !help                                show this message
"""

import re
import select
import sys

from forge import rag, trace
from forge.config import SHOW_DEBUG
from forge.errors import ForgeError
from forge.logger import log
from forge.memory import clear_history, compact_now
from forge.orchestrator import Orchestrator

_FENCE = "```"
_CANCEL = "cancel"


def _stdin_has_data(timeout: float = 0.05) -> bool:
    try:
        return bool(select.select([sys.stdin], [], [], timeout)[0])
    except (OSError, ValueError):
        return False


def _collect_fence_block() -> str | None:
    """
    Read lines until a closing ``` or 'cancel'.
    Returns the collected content, or None if the user cancelled.
    Prints a clear prompt so the user always knows how to exit.
    """
    print(
        "  (paste code, then type ``` on an empty line to submit, or 'cancel' to abort)"
    )
    lines = []
    while True:
        try:
            line = input("... ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if line.strip() == _FENCE:
            break
        if line.strip().lower() == _CANCEL:
            print("[cancelled]\n")
            return None
        lines.append(line)
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    """Remove enclosing ``` markers from a pasted block."""
    text = re.sub(r"^```\w*\n?", "", text, count=1)
    text = re.sub(r"\n?```\s*$", "", text)
    return text


def _read_input() -> str | None:
    """
    Read one user turn.

    Three cases:
    1. Single line → returned as-is.
    2. Fence-first (``` on its own line) → manual code collection.
    3. Multi-line paste → select() detects buffered stdin and
       combines all lines as one turn (fences stripped if present).
    """
    try:
        first_line = input("Forge > ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    # Case 2a: explicit fence on its own line
    if first_line.strip() == _FENCE:
        content = _collect_fence_block()
        return content  # None = cancelled → caller skips

    # Case 2b: question ending with ``` (with optional spaces or lang tag)
    # Matches: "Optimise ce fichier ```"
    #           "Optimise ce fichier ```python"
    #           "Optimise ce fichier ``` "   (trailing space)
    _inline_fence = re.match(r"^(.*?)```\w*\s*$", first_line)
    if _inline_fence:
        question = _inline_fence.group(1).strip()
        content = _collect_fence_block()
        if content is None:
            return question or None
        return f"{question}\n\n{content}".strip() if question else content

    # Case 3: detect a paste (all lines arrive in the buffer at once)
    extra_lines = []
    while _stdin_has_data():
        try:
            extra_lines.append(input("... "))
        except (EOFError, KeyboardInterrupt):
            break

    if not extra_lines:
        return first_line  # Case 1

    # Strip enclosing fences from the pasted block if present,
    # then combine as "question\n\ncode"
    combined = "\n".join(extra_lines)
    code_body = _strip_fences(combined).strip()
    if code_body:
        return f"{first_line}\n\n{code_body}"
    return first_line


def _handle_remember(arg: str) -> None:
    parts = arg.split(maxsplit=2)
    if len(parts) < 3:
        print("Usage: !remember <decision|todo|fact> <project|-> <content>\n")
        return

    kind, project, content = parts
    if kind not in ("decision", "todo", "fact"):
        print("kind must be 'decision', 'todo', or 'fact'\n")
        return
    project = None if project == "-" else project

    conn = rag.get_connection()
    try:
        try:
            entry_id = rag.remember(conn, kind=kind, content=content, project=project)
        except rag.EmbeddingError as e:
            print(f"[remember failed: embedding server unreachable ({e})]\n")
            return
    finally:
        conn.close()
    print(f"[remembered #{entry_id}]\n")


def _handle_recall(arg: str) -> None:
    query = arg.strip()
    if not query:
        print("Usage: !recall <query>\n")
        return

    conn = rag.get_connection()
    try:
        try:
            results = rag.search(conn, query=query, top_k=5)
        except rag.EmbeddingError as e:
            print(f"[recall failed: embedding server unreachable ({e})]\n")
            return
    finally:
        conn.close()

    if not results:
        print("[no matches]\n")
        return

    lines = []
    for r in results:
        proj = f"/{r['project']}" if r["project"] else ""
        lines.append(
            f"  [{r['kind']}{proj}] {r['content']}  (distance={r['distance']:.3f})"
        )
    print("\n".join(lines) + "\n")


def _handle_command(raw: str) -> None:
    # Only the command keyword is case-insensitive -- lowercasing the
    # whole line here would also lowercase !remember/!recall content,
    # which is real text meant to be stored/searched as typed.
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "!clear":
        clear_history()
        print("[context cleared]\n")
    elif cmd == "!compact":
        removed = compact_now()
        if removed:
            print(f"[compacted: {removed} messages summarized]\n")
        else:
            print("[nothing to compact]\n")
    elif cmd == "!trace":
        print(trace.format_for_display(trace.read_last(5)) + "\n")
    elif cmd == "!remember":
        _handle_remember(arg)
    elif cmd == "!recall":
        _handle_recall(arg)
    elif cmd == "!help":
        print(__doc__)
    else:
        print(f"[unknown command: {cmd}] Type !help for the list.\n")


def main():
    orchestrator = Orchestrator()

    print("Forge ready" + (" [debug]" if SHOW_DEBUG else "") + ". Type 'exit' to quit.")
    print(
        "Tip: just type your question. Use ``` only to paste code blocks. !help for more.\n"
    )

    while True:
        user_input = _read_input()

        if user_input is None:
            continue  # None = cancelled fence, not EOF

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit"):
            break
        if stripped.startswith("!"):
            _handle_command(stripped)
            continue

        try:
            result = orchestrator.run(user_input)
        except ForgeError as e:
            log.error("unhandled runtime error: %s", e)
            print(f"\n[error] {e}\n")
            continue

        print("\n" + str(result.output) + "\n")


if __name__ == "__main__":
    main()
