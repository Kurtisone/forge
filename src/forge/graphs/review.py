"""
Forge review graph: read a file, optionally run its tests, send it
to the LLM for analysis.

Chains up to three operations into a reproducible, traceable
pipeline: filesystem read, optional test execution, then LLM
reasoning. The test step only runs when a test_path is supplied --
without one, this behaves exactly like the original review graph
(read_file -> llm_review, 2 steps). With one, test output becomes
primary evidence for the review: a failing test matters more than a
style nitpick (read_file -> run_tests -> llm_review, 3 steps).

This absorbed what was briefly a separate "code" agent
(forge.agents.code) -- on reflection that was the same read+review
flow with one extra step, not a genuinely different concept. Kept as
a single graph rather than a parallel "agent" abstraction: one
vocabulary for the router/API/CLI to reason about, not two that
overlap.

Nodes:
  read_file   — reads the target file, puts its content in
                state.context["file_content"]
  run_tests   — runs `pytest <test_path>` via the dedicated test tool
                (tools/test.py) if test_path is set in context, else
                skipped entirely (the graph does not even visit this
                node -- see the conditional edges in build())
  llm_review  — builds a review prompt (including test output when
                present), calls the LLM, stores the answer

Edges:
  read_file → run_tests   (if ok AND test_path is set)
  read_file → llm_review  (if ok AND no test_path — the plain,
                           backward-compatible path)
  read_file → error       (if file could not be read)
  run_tests → llm_review  (always, once reached — test failures are
                           context for the review, not a hard stop)

Usage (CLI):
  forge review path/to/file.py
  forge review path/to/file.py "focus on security"

Usage (Python):
  from forge.graphs.review import run
  print(run("src/forge/main.py", question="What can be improved?"))
  print(run("src/forge/graph.py", test_path="tests/test_graph.py"))
"""

import json
import re
from pathlib import Path

from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.tools import test as test_tool
from forge.types import AgentState

_MAX_FILE_CHARS = 8_000
_MAX_TEST_OUTPUT_CHARS = 3_000

# A ceiling on how much of a degenerate/garbage response to show,
# mirroring router/parser.py's _MAX_FALLBACK_CHARS -- kept as a
# separate constant since review responses are naturally longer than
# a router tool decision and shouldn't share the same cap.
_MAX_REVIEW_OUTPUT_CHARS = 4_000

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Deliberately duplicated from router/parser.py's _PROMPT_LEAK_MARKERS
# rather than imported -- same reasoning as TOOL_DESCRIPTIONS in
# router/prompt.py: these are underscore-private symbols in a
# different module, and this list only needs to stay in sync with
# what THIS prompt (_REVIEW_PROMPT below) could plausibly leak, not
# with the router's own prompt template.
_PROMPT_LEAK_MARKERS = [
    "Respond in plain text",
    "Be concise and specific",
    "GOOD ANSWER:",
    "NEVER DO THIS",
]

_REVIEW_PROMPT = """/no_think
You are a code reviewer. Analyse the file below and provide clear,
actionable feedback. Focus on: correctness, readability, performance,
and security. If test output is provided, use it as primary evidence
-- a failing test matters more than a style nitpick. Write in the
same language as the question.

File: {filename}
Question: {question}

--- file content ---
{content}
--- end of file ---
{test_section}
Respond in plain text ONLY. Do NOT wrap your answer in JSON, and do
NOT return a {{"tool":...,"content":...}} object -- that format is
for a different system (a routing decision) and never applies here.
Just write your review directly, as a person would speak it.

Example of the expected format, for a small unrelated function:
GOOD ANSWER: The function works but has no input validation -- a
negative value for n would produce an incorrect result. Consider
adding a check at the top and raising ValueError. Naming and
structure are otherwise clear.
NEVER DO THIS: {{"tool":"chat","content":"..."}}
NEVER DO THIS EITHER: {{"tool":"code","content":"..."}}

Now write your own review of the file above, in the same plain
format as GOOD ANSWER -- not the NEVER DO THIS shapes. Be concise
and specific.
"""

_TEST_SECTION_TEMPLATE = """
--- test output ({test_path}) ---
{test_output}
--- end of test output ---
"""


def _clean_review_response(raw: str) -> str:
    """
    Clean the review LLM's plain-text answer.

    Deliberately NOT run through router.parser's JSON-extraction
    cascade: that parser exists to pull a {"tool":...,"content":...}
    decision out of router output, and the review prompt explicitly
    asks for plain text, no JSON. Reusing it here misfired in
    practice -- a small model heavily fine-tuned on the router's JSON
    habit sometimes answers a review prompt with a degenerate JSON
    echo instead of real analysis (e.g.
    {"tool":"chat","content":"hello.go"}), and the router parser
    dutifully "succeeds" at extracting that content as if it were the
    real answer -- silently discarding everything else and producing
    a tiny, plausible-looking but meaningless result. Observed live:
    an 8-character output that was just the reviewed file's name.

    Only <think> blocks and a leaked-prompt echo are stripped here.

    One exception, added after a second real occurrence: if the ENTIRE
    cleaned response is a JSON object shaped like the router's
    {"tool":...,"content":"..."} decision, the "content" field is
    unwrapped and used IF it looks like a substantive answer (>= 8
    words or >= 40 chars) -- calibrated against two real cases: a
    degenerate echo whose "content" was just the reviewed file's name
    (1 word, 8 chars, clearly not a review) versus a case where the
    model wrote a genuine multi-sentence review but still wrapped it
    in the JSON shape despite the prompt's explicit example telling it
    not to. A short/word-poor "content" is NOT trusted and the raw
    JSON is shown as-is instead -- a visibly wrong response beats one
    that's silently and confidently truncated to something that
    happens to look like a valid short answer.
    """
    cleaned = _THINK_BLOCK.sub("", raw).strip()

    unwrapped = _try_unwrap_router_json(cleaned)
    if unwrapped is not None:
        cleaned = unwrapped

    if any(marker in cleaned for marker in _PROMPT_LEAK_MARKERS):
        log.warning("review: model echoed prompt instructions instead of answering")
        return "[error] Le modèle n'a pas généré de réponse exploitable. Réessayez."

    if not cleaned:
        log.warning("review: model returned an empty response")
        return "[error] Le modèle n'a pas généré de réponse. Réessayez."

    if len(cleaned) > _MAX_REVIEW_OUTPUT_CHARS:
        cleaned = cleaned[:_MAX_REVIEW_OUTPUT_CHARS].rstrip() + "…"
        log.warning("review response truncated to %d chars", _MAX_REVIEW_OUTPUT_CHARS)

    return cleaned


_MIN_UNWRAPPED_WORDS = 8
_MIN_UNWRAPPED_CHARS = 40


def _try_unwrap_router_json(cleaned: str) -> str | None:
    """
    If *cleaned* is exactly a {"tool":...,"content":"..."} object (the
    router's decision shape) and "content" looks like a substantive
    answer, return the unwrapped content. Otherwise return None,
    leaving the caller to show the raw text as-is -- see
    _clean_review_response's docstring for the calibration behind the
    substantive/degenerate threshold.
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
    if word_count >= _MIN_UNWRAPPED_WORDS or len(content) >= _MIN_UNWRAPPED_CHARS:
        log.warning(
            "review: model wrapped a substantive answer in router-style JSON "
            "(%d words) despite instructions -- unwrapped it",
            word_count,
        )
        return content.strip()

    log.warning(
        "review: model answered with degenerate JSON-wrapped content %r -- "
        "not trusted as a real answer, showing raw",
        content,
    )
    return None


def _read_file_node(state: AgentState) -> AgentState:
    path_str = state.context.get("file_path", state.user_input.strip())
    path = Path(path_str)

    if not path.exists() or not path.is_file():
        state.ok = False
        state.error = f"File not found: {path_str}"
        state.final_output = f"[error] {state.error}"
        return state

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] cannot read file: {e}"
        return state

    if len(content) > _MAX_FILE_CHARS:
        content = (
            content[:_MAX_FILE_CHARS] + f"\n... (truncated at {_MAX_FILE_CHARS} chars)"
        )
        log.warning("review: file truncated to %d chars", _MAX_FILE_CHARS)

    state.context["file_content"] = content
    state.context["file_name"] = path.name
    log.event("review.read_file", name=path.name, chars=len(content))
    return state


def _run_tests_node(state: AgentState) -> AgentState:
    test_path = state.context.get("test_path")
    if not test_path:
        # Not reached in practice -- build()'s conditional edges skip
        # straight to llm_review when there's no test_path. Kept as a
        # defensive no-op in case this node is ever called directly.
        return state

    # Dedicated test tool (own allowlist/timeout, WORKSPACE_DIR
    # confinement) -- not the general shell tool, see tools/test.py.
    output = test_tool.run(f"pytest {test_path}")
    if len(output) > _MAX_TEST_OUTPUT_CHARS:
        output = (
            output[:_MAX_TEST_OUTPUT_CHARS]
            + f"\n... (truncated at {_MAX_TEST_OUTPUT_CHARS} chars)"
        )

    state.context["test_output"] = output
    # Test failures are information for the review, not a pipeline
    # failure -- state.ok stays True so we proceed to llm_review.
    log.event("review.run_tests", test_path=test_path, output_chars=len(output))
    return state


def _llm_review_node(state: AgentState) -> AgentState:
    content = state.context.get("file_content", "")
    filename = state.context.get("file_name", "unknown")
    question = state.context.get("question", "Que peut-on améliorer ?")
    test_path = state.context.get("test_path")
    test_output = state.context.get("test_output")

    test_section = ""
    if test_path and test_output:
        test_section = _TEST_SECTION_TEMPLATE.format(
            test_path=test_path, test_output=test_output
        )

    prompt = _REVIEW_PROMPT.format(
        filename=filename,
        question=question,
        content=content,
        test_section=test_section,
    )

    log.event("review.llm_call", filename=filename, prompt_chars=len(prompt))
    try:
        raw = call_llm(prompt)
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    # Logged unconditionally (mirrors orchestrator.py's
    # router.raw_output) -- this call previously had no raw-output
    # visibility at all, which made the JSON-habit bug above
    # impossible to confirm from logs alone the first time it happened.
    log.event("review.raw_output", raw=raw)

    state.final_output = _clean_review_response(raw)
    state.final_tool = "review"
    log.event("review.done", chars=len(state.final_output))
    return state


def _error_node(state: AgentState) -> AgentState:
    log.warning("review graph: %s", state.error)
    state.ok = True  # surface as message, not crash
    return state


def build() -> Graph:
    g = Graph("review", max_steps=5)
    g.add_node("read_file", _read_file_node)
    g.add_node("run_tests", _run_tests_node)
    g.add_node("llm_review", _llm_review_node)
    g.add_node("error", _error_node)

    # Order matters -- Graph takes the first matching edge (see
    # graph.py docstring). run_tests is only visited when a
    # test_path was actually given; otherwise this behaves exactly
    # like the original 2-step review graph.
    g.add_edge(
        "read_file",
        "run_tests",
        condition=lambda s: s.ok and bool(s.context.get("test_path")),
    )
    g.add_edge("read_file", "error", condition=lambda s: not s.ok)
    g.add_edge("read_file", "llm_review")  # fallback: ok, no test_path
    g.add_edge("run_tests", "llm_review")  # always, once reached

    return g


def run(
    file_path: str,
    question: str = "Que peut-on améliorer ?",
    test_path: str | None = None,
) -> str:
    """Review a file and return the LLM's analysis as a plain string.

    If test_path is given, its tests are run first and the output is
    fed to the review as primary evidence.
    """
    state = build().run(
        file_path,
        initial_context={
            "file_path": file_path,
            "question": question,
            "test_path": test_path,
        },
    )
    return state.final_output or ""
