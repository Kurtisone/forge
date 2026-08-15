"""
Turns raw router LLM output into a validated RouterDecision.

Extraction cascade (applied in order):
0. Repetition loop guard  ("Allo Allo Allo..." → placeholder)
1. Last valid JSON object  (takes the LAST, not first, complete
   {"tool":...} block — models tend to echo earlier JSON from
   history then generate a better answer at the end)
2. XML <tool_call> block   (Qwen HERETIC fine-tune format)
3. Markdown code fence     (correct code, no JSON wrapper)
4. Leaked-prompt strip     (model echoed its instructions back)
5. Plain-text fallback     (capped at 400 chars to prevent walls
   of leaked JSON + analysis text reaching the user)
"""

import json
import re

from forge.logger import log
from forge.types import RouterDecision

_VALID_TOOLS = {"chat", "code"}


def _valid_tools() -> set[str]:
    """
    The set of tool names the router is allowed to pick, right now.

    Sourced from forge.tools.registry.available_tools() -- the same
    ENABLED_TOOLS-gated set the Graph engine dispatches against -- so
    a tool never becomes routable from conversation just because a
    module happens to exist. _VALID_TOOLS ({"chat", "code"}) is kept
    as the floor: even a misconfigured ENABLED_TOOLS that excludes
    them can't make the router unable to fall back to chat.
    """
    from forge.tools import registry

    return _VALID_TOOLS | set(registry.available_tools())


_LEAKED_ROLE_PREFIX = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)
_XML_CONTENT = re.compile(r"<content>\s*(.*?)\s*</content>", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

# Phrases that only appear in the prompt template, never in a real answer.
# If the model echoes these, it has confused prompt with output.
_PROMPT_LEAK_MARKERS = [
    "No explanation or text outside the JSON",
    "NEVER add text outside the JSON",
    'WHAT "content" MEANS PER TOOL',
    "Stop generating immediately after the closing brace",
    # "they said:" used to live here alongside "you answered:". The
    # history block no longer renders user turns as a bullet -- they are
    # rendered exactly like the live "User:" line so that each prompt is
    # a pure append over the last one (see router/prompt.render_user_turn).
    # "User:" itself is far too generic to use as a leak marker, so the
    # replacement is the history header, which is template-only text and
    # cannot plausibly appear in a real answer.
    "you answered:",
    "is the new message you must answer now",
]

# Max chars shown to the user for a plain-text fallback.
# Beyond this the content is almost certainly noise (leaked JSON,
# multi-paragraph analysis, etc.).
_MAX_FALLBACK_CHARS = 400


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def _contains_leaked_prompt(text: str) -> bool:
    return any(marker in text for marker in _PROMPT_LEAK_MARKERS)


def _extract_xml_content(text: str) -> str | None:
    m = _XML_CONTENT.search(text)
    return m.group(1).strip() if m else None


def _extract_code_fence(text: str) -> str | None:
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else None


def _strip_leaked_role_prefix(text: str) -> str:
    return _LEAKED_ROLE_PREFIX.sub("", text, count=1)


def _matching_brace(text: str, start: int) -> int | None:
    """
    Index of the "}" closing the "{" at *start*, or None if it never
    closes.

    Braces inside JSON string literals are skipped, which is the whole
    point of this function existing rather than a plain counter: a
    router object's "content" carries a nested JSON payload whose own
    "content" is arbitrary file text, and real file text has braces
    that don't balance (a Go/C/Rust/JS snippet cut mid-function, a
    stray "}" in a comment, a Python dict literal). A blind counter
    hit zero early -- or never -- and threw away a perfectly valid
    router decision, sending files:write to the plain-text fallback.
    "print('...')" happened to survive only because it has no brace
    at all.
    """
    depth = 0
    in_string = False
    escaped = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
    return None


def _all_json_objects(text: str) -> list[dict]:
    """
    Return all complete, parseable {"tool":..., "content":...} objects
    found in text, in order of appearance.
    """
    results = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start == -1:
            break
        end = _matching_brace(text, start)
        if end is None:
            # This "{" never closes -- a truncated object, or a brace
            # inside prose the model wrote around its JSON. Resume the
            # search one char later instead of giving up on the rest of
            # the text: a later object can still be complete, and with
            # depth counting it would otherwise be unreachable (its
            # closing brace only ever brings depth 2 -> 1, never 0).
            i = start + 1
            continue
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "tool" in obj:
            results.append(obj)
        i = end + 1
    return results


def _is_repetition_loop(text: str, threshold: float = 0.6) -> bool:
    tokens = text.split()
    if len(tokens) < 10:
        return False
    most_common = max(set(tokens), key=tokens.count)
    return tokens.count(most_common) / len(tokens) > threshold


def _validate_json_obj(data: dict, cleaned: str) -> RouterDecision | None:
    """
    Turn a parsed JSON dict into a RouterDecision if it looks valid,
    or return None to try the next extraction step.
    """
    tool = data.get("tool", "chat")
    content = data.get("content")
    if tool not in _valid_tools():
        log.warning("router picked unknown tool %r, falling back to chat", tool)
        tool = "chat"
    if not content or (isinstance(content, str) and not content.strip()):
        return None  # empty content → try next extraction

    # A tool whose payload is itself JSON (files, memory, review,
    # sysadmin) may send it as a real nested object rather than a
    # JSON string, and that's now the shape the prompt teaches. It
    # removes a whole level of escaping the model demonstrably does
    # not hold: nesting JSON inside a JSON *string* needs \\n where
    # the model writes \n, and the inner parse then dies on an
    # invalid control character -- which is what made files:write
    # fail for any multi-line file body. Re-encoding here keeps
    # RouterDecision.content a string and every tool's run(content)
    # contract untouched: they still receive JSON text to parse, just
    # text this module produced instead of the model.
    if isinstance(content, dict | list):
        content = json.dumps(content, ensure_ascii=False)
    if not str(content).strip():
        return None
    # Optional multi-step continuation flag. Absent (the common case,
    # and every fine-tune/model that predates this field) means True:
    # one step, same as before. Only an explicit false continues the
    # loop in the orchestrator.
    done = bool(data.get("done", True))
    return RouterDecision(tool=tool, content=str(content), raw=cleaned, done=done)


def parse_router_output(raw: str) -> RouterDecision:
    cleaned = _strip_think_blocks(raw)

    # 0. Repetition loop guard
    if _is_repetition_loop(cleaned):
        log.warning("router output is a repetition loop, returning placeholder")
        return RouterDecision(
            tool="chat",
            content="Je n'ai pas pu générer une réponse utile. Reformulez ou réessayez.",
            raw=raw,
            is_fallback=True,
        )

    # 1. JSON — try the LAST valid object first, fall back to first.
    #    When a model echoes history + generates a new answer, the last
    #    JSON object is the intended response; the earlier ones are noise.
    json_objects = _all_json_objects(cleaned)
    for data in reversed(json_objects):
        decision = _validate_json_obj(data, cleaned)
        if decision:
            return decision

    # 2. XML tool-call
    xml_content = _extract_xml_content(cleaned)
    if xml_content:
        log.warning("router used XML tool-call format, extracting <content>")
        return RouterDecision(tool="chat", content=xml_content, raw=raw)

    # 3. Markdown code fence
    code_content = _extract_code_fence(cleaned)
    if code_content:
        log.warning("router returned a markdown code block, routing to code tool")
        return RouterDecision(tool="code", content=code_content, raw=raw)

    # 4. Plain-text fallback
    log.warning("router returned non-JSON output, falling back to chat")

    fallback = _strip_leaked_role_prefix(cleaned.strip())

    # If the output leaked prompt instructions, it's noise, not an answer.
    if _contains_leaked_prompt(fallback):
        log.warning(
            "router output contains leaked prompt instructions, returning placeholder"
        )
        return RouterDecision(
            tool="chat",
            content="Je n'ai pas pu générer une réponse. Réessayez.",
            raw=raw,
            is_fallback=True,
        )

    if not fallback:
        log.warning("router output was empty, returning placeholder")
        return RouterDecision(
            tool="chat",
            content="Je n'ai pas pu générer une réponse. Réessayez.",
            raw=raw,
            is_fallback=True,
        )

    # Cap length: anything beyond _MAX_FALLBACK_CHARS is almost certainly
    # a mix of leaked JSON + analysis text — not a useful answer.
    if len(fallback) > _MAX_FALLBACK_CHARS:
        fallback = fallback[:_MAX_FALLBACK_CHARS].rstrip() + "…"
        log.warning("fallback content truncated to %d chars", _MAX_FALLBACK_CHARS)

    return RouterDecision(tool="chat", content=fallback, raw=raw)
