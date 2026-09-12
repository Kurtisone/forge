"""
Pull the web UI's renderer out of index.html so it can be executed.

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

    scheme = re.search(r"^const _SAFE_LINK_SCHEME = .*$", src, re.M)
    assert scheme, "_SAFE_LINK_SCHEME vanished from index.html"

    parts = [scheme.group(0)]
    for name in WANTED:
        m = re.search(rf"^(?:const|function) {name}\b", src, re.M)
        assert m, f"{name} vanished from index.html"
        parts.append(_balanced(src, m.start()))

    parts.append("export { formatContent, inlineMarkdown, escapeHtml };")
    return "\n\n".join(parts) + "\n"
