"""
Pull pieces of the web UI out of index.html so they can be executed.

Three test files in this suite open with the same caveat -- "this suite
has no JS runtime, so these are assertions on the source; a tripwire,
not a proof". That was true of CI and never true of the machine Forge
is written on, which has deno. The first time the renderer was actually
RUN it turned up a bug that had been on screen for weeks: `_like this_`
reaches the user with its underscores showing, because inlineMarkdown
implements `**bold**` and `*em*` and nothing else.

Extraction rather than a copy, deliberately. A second copy of these
functions in a test fixture is a copy that drifts, and drift in a
renderer is invisible until someone reads their own output and finds
underscores in it.
"""

from __future__ import annotations

import re
from pathlib import Path

WANTED = ("escapeHtml", "formatContent", "inlineMarkdown", "renderDiff")

#: Module-level allowlists the functions above close over. Named
#: rather than pattern-matched: a missing one is a ReferenceError
#: at render time, which reads as a broken test rather than as a
#: constant that moved.
CONSTANTS = ("_SAFE_LINK_SCHEME", "_SAFE_IMAGE_SRC")


def _balanced(src: str, start: int) -> str:
    """From `start` to the brace that closes the first one opened."""
    depth, i, opened = 0, start, False
    while i < len(src):
        if src[i] == "{":
            depth += 1
            opened = True
        elif src[i] == "}":
            depth -= 1
            if opened and depth == 0:
                return src[start : i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces from offset {start}")


def renderer_module(index_html: Path) -> str:
    """An ES module exporting the UI's own formatContent."""
    src = index_html.read_text(encoding="utf-8")

    parts = []
    for name in CONSTANTS:
        m = re.search(rf"^const {name} = .*$", src, re.MULTILINE)
        assert m, f"{name} vanished from index.html"
        parts.append(m.group(0))

    for name in WANTED:
        m = re.search(rf"^(?:const|function) {name}\b", src, re.MULTILINE)
        assert m, f"{name} vanished from index.html"
        parts.append(_balanced(src, m.start()))

    parts.append("export { formatContent, inlineMarkdown, escapeHtml };")
    return "\n\n".join(parts) + "\n"


def commands_module(index_html: Path) -> str:
    """
    An ES module exporting the UI's own `!` command table.

    Every message starting with `!` is handled in the browser and never
    reaches the server, so this table is the whole of what those
    commands do -- and a command missing from it is not a degraded
    command, it is an error message. That was `!pair`: intercepted
    server-side, unreachable from the one interface it was written for.

    `apiFetch` is left undeclared on purpose. The test injects its own,
    which is what makes it possible to assert on the request a command
    actually sends rather than on the source that composes it.
    """
    src = index_html.read_text(encoding="utf-8")

    m = re.search(r"^const UI_COMMANDS\b", src, re.MULTILINE)
    assert m, "UI_COMMANDS vanished from index.html"

    return (
        "export function commands(apiFetch, confirm) {\n"
        + _balanced(src, m.start())
        + "\n  return UI_COMMANDS;\n}\n"
    )
