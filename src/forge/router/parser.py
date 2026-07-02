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
_LEAKED_ROLE_PREFIX = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)
_XML_CONTENT = re.compile(r"<content>\s*(.*?)\s*</content>", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

# Phrases that only appear in the prompt template, never in a real answer.
# If the model echoes these, it has confused prompt with output.
_PROMPT_LEAK_MARKERS = [
    "No explanation or text outside the JSON",
    "NEVER add text outside the JSON",
    "WHAT \"content\" MEANS PER TOOL",
    "Stop generating immediately after the closing brace",
    "they said:",
    "you answered:",
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
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : j + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "tool" in obj:
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            break
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
    if tool not in _VALID_TOOLS:
        log.warning("router picked unknown tool %r, falling back to chat", tool)
        tool = "chat"
    if not content or not str(content).strip():
        return None   # empty content → try next extraction
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
        log.warning("router output contains leaked prompt instructions, returning placeholder")
        return RouterDecision(
            tool="chat",
            content="Je n'ai pas pu générer une réponse. Réessayez.",
            raw=raw,
        )

    if not fallback:
        log.warning("router output was empty, returning placeholder")
        return RouterDecision(
            tool="chat",
            content="Je n'ai pas pu générer une réponse. Réessayez.",
            raw=raw,
        )

    # Cap length: anything beyond _MAX_FALLBACK_CHARS is almost certainly
    # a mix of leaked JSON + analysis text — not a useful answer.
    if len(fallback) > _MAX_FALLBACK_CHARS:
        fallback = fallback[:_MAX_FALLBACK_CHARS].rstrip() + "…"
        log.warning("fallback content truncated to %d chars", _MAX_FALLBACK_CHARS)

    return RouterDecision(tool="chat", content=fallback, raw=raw)
