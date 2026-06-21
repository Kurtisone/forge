"""
Turns raw router LLM output into a validated RouterDecision.

A malformed JSON response does NOT crash the run -- it falls back to
"chat" with the raw text, same behavior as before -- but it is now a
typed object instead of a bare dict, and the fallback is logged
instead of silent.
"""

import json
import re

from forge.logger import log
from forge.types import RouterDecision

_VALID_TOOLS = {"chat", "code"}
_LEAKED_ROLE_PREFIX = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)


def _strip_leaked_role_prefix(text: str) -> str:
    """
    Local models occasionally leak a role label ('Assistant: ...')
    instead of staying in pure JSON. This is defense in depth on top
    of the prompt fix -- it should rarely trigger, but if it does,
    the label shouldn't end up visible in the final answer.
    """
    return _LEAKED_ROLE_PREFIX.sub("", text, count=1)


def _extract_first_json_object(text: str):
    """
    Find and parse the first balanced {...} object in text.

    Local models sometimes keep generating after the JSON object
    (a hallucinated new "User: ..." turn, repeated examples, etc.).
    Requiring the *entire* string to be valid JSON throws away a
    perfectly good answer just because of trailing noise, so we scan
    for the first balanced brace pair instead.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def parse_router_output(raw: str) -> RouterDecision:
    data = _extract_first_json_object(raw)

    if not isinstance(data, dict):
        log.warning("router returned non-JSON output, falling back to chat")
        return RouterDecision(
            tool="chat", content=_strip_leaked_role_prefix(raw.strip()), raw=raw
        )

    tool = data.get("tool", "chat")
    content = data.get("content")

    if tool not in _VALID_TOOLS:
        log.warning("router picked unknown tool %r, falling back to chat", tool)
        tool = "chat"

    if not content or not str(content).strip():
        log.warning("router returned empty content, falling back to chat")
        tool = "chat"
        content = raw.strip()

    return RouterDecision(tool=tool, content=str(content), raw=raw)
