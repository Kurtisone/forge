"""
Shared cleaning helpers for LLM responses that are expected to be
plain text, not JSON.

Extracted after the same bug appeared twice independently: both
graphs/review.py and graphs/research.py prompt this model for plain
prose (not a router-style tool decision), and both fight the same
habit -- the model sometimes wraps an otherwise-good answer in
{"tool":...,"content":"..."} despite explicit instructions and a
worked example not to. review.py's fix (conditionally unwrap when
"content" looks substantive, show the raw JSON as-is otherwise) was
proven live, but research.py was written afterward with its own
copy of the *think-block-and-leak-marker* stripping and no unwrap
logic at all -- the exact bug resurfaced in production on the first
real research run. Centralizing the unwrap logic here means a future
threshold tweak (or another caller) only has one place to change,
instead of drifting silently the way these two just did.

Prompt-specific leaked-instruction markers are NOT centralized here
-- those are tied to each prompt's own wording (review's vs
research's), not shared behavior, and stay defined next to each
prompt per the same reasoning as TOOL_DESCRIPTIONS in
router/prompt.py.
"""

import json
import re

from forge.logger import log

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Calibrated against two real cases seen live in graphs/review.py: a
# degenerate echo whose "content" was just a filename (1 word, 8
# chars) versus a genuine multi-sentence answer that was still wrapped
# in the JSON shape despite instructions not to.
MIN_UNWRAP_WORDS = 8
MIN_UNWRAP_CHARS = 40


def strip_think_blocks(raw: str) -> str:
    return THINK_BLOCK.sub("", raw).strip()


def try_unwrap_router_json(cleaned: str, source: str) -> str | None:
    """
    If *cleaned* is exactly a {"tool":...,"content":"..."} object (the
    router's decision shape) and "content" looks like a substantive
    answer, return the unwrapped content. Otherwise return None,
    leaving the caller to show the raw text as-is -- a visibly wrong
    response beats one silently truncated to something that happens
    to look like a valid short answer.

    *source* is only used to label the warning log (e.g. "review",
    "research") so the two callers stay distinguishable in logs.
    """
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or "content" not in data:
        return None

    content = data.get("content")
    if not isinstance(content, str):
        return None

    word_count = len(content.split())
    if word_count >= MIN_UNWRAP_WORDS or len(content) >= MIN_UNWRAP_CHARS:
        log.warning(
            "%s: model wrapped a substantive answer in router-style JSON "
            "(%d words) despite instructions -- unwrapped it",
            source,
            word_count,
        )
        return content.strip()

    log.warning(
        "%s: model answered with degenerate JSON-wrapped content %r -- "
        "not trusted as a real answer, showing raw",
        source,
        content,
    )
    return None
