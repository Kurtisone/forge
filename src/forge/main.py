"""
Forge REPL.

Single-line usage: just type and press Enter as usual.

Multi-line / code paste:

  A) Start a line with ``` (fence-first):
     Forge > ```
     ... def hello():
     ...     print("hi")
     ... ```

  B) Type your question, then paste the code on the next lines.
     Forge > Create a multi-stage build
     ... ```
     ... FROM python:3.12
     ... ```
     Forge will automatically combine the question and the code into
     a single turn when you paste them together.

Special commands (prefix with !):
  !clear   wipe conversation history so the next turn starts fresh
  !help    show this message
"""

import select
import sys

from forge.config import SHOW_DEBUG
from forge.errors import ForgeError
from forge.logger import log
from forge.memory import clear_history
from forge.orchestrator import Orchestrator

_FENCE = "```"


def _stdin_has_data(timeout: float = 0.05) -> bool:
    """
    Return True if stdin has data buffered right now (i.e. the user
    pasted a block of lines, all of which arrived at once).
    Uses select() which is available on Linux / Steam Deck.
    Falls back to False on platforms that don't support it.
    """
    try:
        return bool(select.select([sys.stdin], [], [], timeout)[0])
    except (OSError, ValueError):
        return False


def _collect_fence_block() -> str:
    """
    Read lines until a closing ``` fence or EOF.
    Returns the code content (without the fence markers).
    """
    lines = []
    while True:
        try:
            line = input("... ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == _FENCE:
            break
        lines.append(line)
    return "\n".join(lines)


def _read_input() -> str | None:
    """
    Read one user turn. Returns the text to submit, or None on EOF.

    Handles three cases:
    1. Single line  →  returned as-is.
    2. Fence-first  →  collects code block, returns it alone.
    3. Question then pasted code  →  detects buffered stdin with
       select(), reads ahead, combines question + code as one turn.
       This is the common copy-paste pattern:
         "Create a multi-stage build\n```\nFROM...\n```"
    """
    try:
        first_line = input("Forge > ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    # Case 2: fence-first
    if first_line.strip() == _FENCE:
        code = _collect_fence_block()
        return code

    # Case 3: question followed immediately by more buffered lines
    # (i.e. the user pasted a question + a code block in one go).
    extra_lines = []
    while _stdin_has_data():
        try:
            line = input("... ")
        except (EOFError, KeyboardInterrupt):
            break
        extra_lines.append(line)

    if not extra_lines:
        # Case 1: plain single line
        return first_line

    # Combine: strip enclosing ``` fences from the extra block,
    # then glue question + code as one readable message.
    combined_extra = "\n".join(extra_lines)
    # Remove leading/trailing fence markers so the prompt stays clean
    import re
    code_body = re.sub(r"^```\w*\n?", "", combined_extra, count=1)
    code_body = re.sub(r"\n?```\s*$", "", code_body)

    if code_body.strip():
        return f"{first_line}\n\n{code_body.strip()}"
    return first_line


def _handle_command(cmd: str) -> bool:
    """Handle a !command. Returns True if the REPL should exit."""
    if cmd == "!clear":
        clear_history()
        print("[context cleared]\n")
    elif cmd == "!help":
        print(__doc__)
    return False


def main():
    orchestrator = Orchestrator()

    print("Forge ready" + (" [debug]" if SHOW_DEBUG else "") + ". Type 'exit' to quit.\n")
    print("Tip: wrap multi-line pastes in ``` fences.  !help for commands.\n")

    while True:
        user_input = _read_input()

        if user_input is None:
            break

        stripped = user_input.strip()

        if not stripped:
            continue

        if stripped.lower() in ("exit", "quit"):
            break

        if stripped.startswith("!"):
            _handle_command(stripped.lower())
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
