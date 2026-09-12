"""
Turns raw router LLM output into a validated RouterDecision.

Extraction cascade (applied in order):
0. Repetition loop guard  ("Allo Allo Allo..." → placeholder).
   Steps 0 and 4 also apply INSIDE a valid JSON object's
   content, which is the path the grammar guarantees and the
   one that used to be unguarded.
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
from collections import Counter

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
    # "No explanation or text outside the JSON" used to head this list
    # and was DEAD: the prompt says "NEVER add text outside the JSON"
    # and has for some time, so the marker could not fire on any output
    # any model could produce. Nothing failed -- a leak marker that
    # matches nothing is silent by nature. test_parser_leak_markers.py
    # now asserts every marker here appears verbatim in a prompt this
    # code actually builds, which is the only direction of this drift a
    # test can close.
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
    # The search-chaining instruction, added 2026-09-12 because it is
    # the one that actually leaked. LFM2.5-8B-A1B answered fixture e02
    # with this sentence and the two after it, verbatim, inside a valid
    # JSON envelope -- so it passed the grammar, passed the cascade, and
    # would have been spoken to the user as the answer to their
    # question. It was in no version of this list.
    #
    # The other direction stays open and is worth naming rather than
    # implying otherwise: this is a closed set with no way to discover
    # its own members, the same shape of gap forge/non_answer.py
    # measured at 22 refusals recognised out of 22 missed. Every
    # sentence in the prompt is a candidate; six are registered.
    "The search results above already contain titles",
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


def _is_repetition_loop(text: str, threshold: float = 0.6, max_ngram: int = 6) -> bool:
    """
    Whether one repeating unit covers most of the text.

    The unit is up to `max_ngram` words, not one word, and that is the
    whole point. Measured on 2026-09-12 against LFM2.5-8B-A1B, which
    produced "les dernieres mises a jour de mises a jour de mises a
    jour de ..." -- a textbook loop that the single-token version of
    this function could not see and never could have. A share of ONE
    token is bounded by the length of the repeating unit: a two-word
    loop caps that share at 50%, a three-word loop at 33%, a four-word
    loop at 25%. Against a 0.6 threshold every loop longer than one
    word was invisible by construction, and the arithmetic says so
    without needing a model to demonstrate it.

    n=1 reproduces the old behaviour exactly, so nothing that was
    caught before stops being caught.
    """
    tokens = text.split()
    if len(tokens) < 10:
        return False
    for n in range(1, max_ngram + 1):
        # Three repeats is the fewest that distinguishes a loop from a
        # writer making a point twice.
        if len(tokens) < n * 3:
            break
        grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        _, count = grams.most_common(1)[0]
        if count >= 3 and count * n / len(tokens) > threshold:
            return True
    return False


def _unwrap_envelope(text: str) -> tuple[str, str | None]:
    """
    A `chat` answer that is itself a router envelope, and what to do
    with it.

    Returns ("answer", text) when the text is a real answer,
    ("unwrap", inner) when it is a complete envelope wrapped one level
    too deep and the inner content is the answer, and ("broken", None)
    when it is an envelope the model never finished -- which is not an
    answer under any reading.

    Both shapes reach the user as a wall of raw JSON without this.
    Measured 2026-09-12 on LFM2.5-8B-A1B: four replies out of
    thirty-six, every one of them truncated. The escaping is why the
    cascade cannot catch them by itself -- an inner envelope sits
    inside a JSON *string*, so its quotes are escaped and
    _all_json_objects() scanning the raw text does not see valid JSON
    there. Only the outer object parses, and its content is the mess.

    Across 72 recorded replies from two models, no real answer opened
    a brace and named both protocol keys. An answer that merely
    mentions them, or that is legitimately JSON data, does neither.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return ("answer", text)
    if '"tool"' not in stripped or '"content"' not in stripped:
        return ("answer", text)
    try:
        data = json.loads(stripped)
    except ValueError:
        return ("broken", None)
    if not isinstance(data, dict) or "content" not in data or "tool" not in data:
        return ("answer", text)
    inner = data["content"]
    if isinstance(inner, dict | list):
        inner = json.dumps(inner, ensure_ascii=False)
    inner = str(inner)
    return ("unwrap", inner) if inner.strip() else ("broken", None)


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

    # The same two checks the plain-text path applies, applied here
    # too -- because this is the path the GBNF grammar guarantees the
    # model takes, and until 2026-09-12 it was the only path with no
    # checks on it at all. Proved with one string: the prompt's own
    # "No explanation or text outside the JSON" served as bare text
    # comes back as a placeholder, and served inside a valid
    # {"tool":"chat","content":...} came back untouched, to be spoken
    # to the user as an answer. The shape the grammar guarantees was
    # exactly the shape that skipped the guards.
    #
    # Found by swapping the served model rather than by reading this
    # file: a 9B produced neither failure in 36 fixtures, LFM2.5-8B-A1B
    # produced nine. A check nothing exercises is not known to work.
    #
    # None rather than a placeholder: an object that leaked is not a
    # decision, and the cascade should go on looking. If every object
    # leaks, step 4 reaches the same text and returns the placeholder,
    # so the outcome is unchanged and the earlier objects still get
    # their chance.
    if _contains_leaked_prompt(str(content)):
        log.warning("router JSON content leaked prompt instructions, skipping")
        return None

    # The repetition guard applies to `chat` and to nothing else. A
    # tool payload is DATA -- a config file of near-identical lines, a
    # CSV, a fixture -- and "this text repeats itself" is a statement
    # about prose degenerating, not about a file being boring. Caught
    # by tests/test_orchestrator.py, whose 200-identical-line fixture
    # this rejected on the first run; the old placement never saw it
    # because json.dumps escapes the newlines, leaving the whole
    # payload as one whitespace-free token.
    # Terminal where it fires, unlike the leak check above it, and the
    # difference is what step 4 can still see. A leaked marker is in
    # the raw text too, so handing the cascade a None ends at the same
    # placeholder either way. A loop is NOT: the envelope around it
    # dilutes the repeating unit below the threshold, so returning
    # None here served the whole raw JSON object to the user as prose
    # -- measured on the real g02 output, where the loop is caught
    # inside `content` and invisible one level out.
    if tool == "chat" and _is_repetition_loop(str(content)):
        log.warning("router JSON content is a repetition loop, discarding")
        return RouterDecision(
            tool="chat",
            content="Je n'ai pas pu générer une réponse utile. Reformulez ou réessayez.",
            raw=cleaned,
            is_fallback=True,
        )
    if tool == "chat":
        # Bounded, because an envelope inside an envelope inside an
        # envelope is the same failure and a `while True` on model
        # output is how a parser hangs.
        for _ in range(3):
            verdict, unwrapped = _unwrap_envelope(str(content))
            if verdict == "answer":
                break
            if verdict == "broken":
                log.warning("router content is an unfinished envelope, discarding")
                return RouterDecision(
                    tool="chat",
                    content=(
                        "Je n'ai pas pu générer une réponse utile. "
                        "Reformulez ou réessayez."
                    ),
                    raw=cleaned,
                    is_fallback=True,
                )
            log.warning("router wrapped its answer in a second envelope, unwrapping")
            content = unwrapped
        else:
            # Budget exhausted and still an envelope. Serving what is
            # left would hand the user the JSON this loop exists to
            # remove, so the giving-up path refuses like the truncated
            # one rather than falling through.
            log.warning("router nested its envelope past any useful depth")
            return RouterDecision(
                tool="chat",
                content=(
                    "Je n'ai pas pu générer une réponse utile. Reformulez ou réessayez."
                ),
                raw=cleaned,
                is_fallback=True,
            )
    # Optional multi-step continuation flag. Absent (the common case,
    # and every fine-tune/model that predates this field) means True:
    # one step, same as before. Only an explicit false continues the
    # loop in the orchestrator.
    done = bool(data.get("done", True))
    return RouterDecision(tool=tool, content=str(content), raw=cleaned, done=done)


def parse_router_output(raw: str) -> RouterDecision:
    cleaned = _strip_think_blocks(raw)

    # 0. Repetition loop guard, on the RAW output, where prose and a
    #    tool payload are not yet distinguishable -- so only the
    #    unambiguous case is rejected here: one token, over and over.
    #    A files:write body of fifty near-identical config lines is a
    #    phrase-level loop by any measure and a perfectly good file,
    #    and it arrives at this point looking exactly like degenerate
    #    prose. The phrase-level test is applied twice below instead,
    #    at the two points where the text is known to be an ANSWER:
    #    inside a validated `chat` object, and on the plain-text
    #    fallback.
    if _is_repetition_loop(cleaned, max_ngram=1):
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

    # The fallback IS prose -- there is no tool payload left to
    # confuse it with -- so the phrase-level test applies here.
    if _is_repetition_loop(fallback):
        log.warning("router fallback text is a repetition loop, returning placeholder")
        return RouterDecision(
            tool="chat",
            content="Je n'ai pas pu générer une réponse utile. Reformulez ou réessayez.",
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
