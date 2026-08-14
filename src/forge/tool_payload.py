"""
Shared JSON loader for tool payloads (files, memory, review,
sysadmin).

Every one of these tools takes its instruction as JSON text and calls
json.loads on it. The router now hands them text it produced itself
(router/parser.py re-encodes a nested object), so the well-formed
path is the normal one. This module covers what's left: text the
model wrote directly.

That happens whenever the object shape isn't in force -- grammar
disabled, a provider without GBNF sampling, an older fine-tune, or
plain model drift back to the escaped-string shape its own history
is full of. In that shape the payload sits inside a JSON *string*,
so every newline in it needs \\\\n; the model writes \\n; the outer
parse then yields inner text carrying a raw newline and strict
json.loads dies on "Invalid control character". The payload is
otherwise perfectly good -- only its escaping is wrong, and only in
a way that is unambiguous to recover from.

json.loads(strict=False) accepts raw control characters inside
strings and is exactly that recovery. It is deliberately a SECOND
attempt, never the default: strict parsing succeeding tells us the
model produced correct JSON, and losing that signal would hide the
next escaping regression instead of surfacing it. The warning log is
the point -- a run where this fires still worked, but the model is
drifting.

Centralized rather than fixed in files.py alone, where the bug was
actually observed: the same failure applies verbatim to the other
three, and this repo has already been bitten twice by fixing one
copy of a shared behavior and leaving its twin to diverge silently
(review.py vs research.py, see text_cleaning.py).
"""

import json

from forge.logger import log


def loads_payload(content: str, tool: str) -> object:
    """
    Parse a tool payload, tolerating under-escaped control characters.

    Raises the same exceptions json.loads does, so callers keep their
    existing error handling unchanged.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        parsed = json.loads(content, strict=False)
        # Only reached when the lenient pass succeeds; a genuinely
        # malformed payload re-raises from here and the caller's own
        # error message stands.
        log.warning(
            "%s payload had unescaped control characters, parsed leniently", tool
        )
        log.event("tool.payload_lenient", tool=tool)
        return parsed
