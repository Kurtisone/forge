"""
Turns raw router LLM output into a validated RouterDecision.

The parser handles several real-world local-model quirks, applied
in order before the final "chat with raw text" fallback:
1. JSON (or JSON with trailing noise / runaway generation).
2. <tool_call> XML block (Qwen HERETIC and similar fine-tunes).
3. Fenced markdown code block (model answered with correct code but
   skipped the JSON wrapper entirely).
4. <think>...</think> stripping (Qwen3 chain-of-thought).
5. Leaked "Assistant:" role prefix.
"""

import json
import re

from forge.logger import log
from forge.types import RouterDecision

_VALID_TOOLS = {"chat", "code"}
_LEAKED_ROLE_PREFIX = re.compile(r"^\s*(assistant|user)\s*:\s*", re.IGNORECASE)
_XML_CONTENT = re.compile(r"<content>\s*(.*?)\s*</content>", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# First complete fenced code block, optional language tag
_CODE_FENCE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def _extract_xml_content(text: str) -> str | None:
    m = _XML_CONTENT.search(text)
    return m.group(1).strip() if m else None


def _extract_code_fence(text: str) -> str | None:
    """
    If the model responded with a fenced code block instead of JSON,
    extract the code inside the first ``` ... ``` pair.
    Handles the common case where a local model produces the correct
    answer as markdown rather than a JSON-wrapped string.
    """
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else None


def _strip_leaked_role_prefix(text: str) -> str:
    return _LEAKED_ROLE_PREFIX.sub("", text, count=1)


def _extract_first_json_object(text: str):
    """
    Find and parse the first balanced {...} object in text, tolerant
    of trailing noise after the closing brace.
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
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_router_output(raw: str) -> RouterDecision:
    cleaned = _strip_think_blocks(raw)

    # 1. JSON
    data = _extract_first_json_object(cleaned)
    if isinstance(data, dict):
        tool = data.get("tool", "chat")
        content = data.get("content")
        if tool not in _VALID_TOOLS:
            log.warning("router picked unknown tool %r, falling back to chat", tool)
            tool = "chat"
        if not content or not str(content).strip():
            log.warning("router returned empty content, falling back to chat")
            tool, content = "chat", cleaned.strip()
        return RouterDecision(tool=tool, content=str(content), raw=raw)

    # 2. XML tool-call
    xml_content = _extract_xml_content(cleaned)
    if xml_content:
        log.warning("router used XML tool-call format, extracting <content>")
        return RouterDecision(tool="chat", content=xml_content, raw=raw)

    # 3. Markdown code fence — correct code, wrong format
    code_content = _extract_code_fence(cleaned)
    if code_content:
        log.warning("router returned a markdown code block, routing to code tool")
        return RouterDecision(tool="code", content=code_content, raw=raw)

    # 4. Plain text fallback
    log.warning("router returned non-JSON output, falling back to chat")
    fallback = _strip_leaked_role_prefix(cleaned.strip())
    if not fallback:
        log.warning("router output was empty after cleanup, returning placeholder")
        fallback = "I could not generate a response. Please try again."
    return RouterDecision(tool="chat", content=fallback, raw=raw)
