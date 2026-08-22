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

from pathlib import Path

from forge import lang, subtrace
from forge.config import ENFORCE_ANSWER_LANGUAGE
from forge.context_info import today_line
from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.text_cleaning import (
    looks_like_a_copy,
    strip_think_blocks,
    try_unwrap_router_json,
)
from forge.tools import test as test_tool
from forge.types import AgentState

_MAX_FILE_CHARS = 8_000
_MAX_TEST_OUTPUT_CHARS = 3_000

# Said three times, to three different readers, because each one drew
# the wrong conclusion on its own:
#   _CUT_MARKER       -- to the model, at the cut, in Forge's voice
#   _TRUNCATION_NOTE  -- to the model, near the question, as a rule
#   _TRUNCATION_FOOTER-- to the user, in the answer, deterministically
# The first two are wording and wording has lost seven times here. The
# third is the one that cannot fail: a review of the first 8 000
# characters of a 40 000-character file is not a review of that file,
# and until now nothing said so anywhere the user could see it.
_CUT_MARKER = (
    "\n[cut by Forge at {shown} of {total} characters -- the rest was not sent]"
)

_TRUNCATION_NOTE = """
NOTE: this file is {total} characters long and Forge sent you only the
first {shown}. The text therefore stops mid-way. That cut is Forge's,
not the author's: never report the abrupt ending, a missing closing
section, or an apparent copy-paste error as a defect of the file, and
never guess what the unsent part contains.
"""

_TRUNCATION_FOOTER = (
    "\n\n---\n_Revue partielle : Forge n'a analysé que les {shown} "
    "premiers caractères sur {total} ({percent} % du fichier)._"
)

# A ceiling on how much of a degenerate/garbage response to show,
# mirroring router/parser.py's _MAX_FALLBACK_CHARS -- kept as a
# separate constant since review responses are naturally longer than
# a router tool decision and shouldn't share the same cap.
_MAX_REVIEW_OUTPUT_CHARS = 4_000

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

# The "/no_think" prefix below is NOT dead, however dead it looks.
# Qwen3.5 dropped the /think soft switch, and the router GBNF grammar
# (applied to every call, not just routing) already makes a reasoning
# block impossible -- so on paper the token buys nothing. Measured on
# 2026-08-16 with bench/no_think_ab.py, removing it made this model
# return the GOOD ANSWER example below instead of a real answer, twice,
# deterministically. Whatever it does at position 0 is not what its
# name says. Run that harness before touching it.
_REVIEW_PROMPT = """/no_think
{today_line}
You are a code reviewer. Analyse the file below and provide clear,
actionable feedback. Focus on: correctness, readability, performance,
and security. If test output is provided, use it as primary evidence
-- a failing test matters more than a style nitpick. Write in the
same language as the question.

File: {filename}
Question: {question}
{truncation_note}
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


def _clean_review_response(raw: str, file_content: str = "") -> str:
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
    cleaned = strip_think_blocks(raw)

    unwrapped = try_unwrap_router_json(cleaned, source="review")
    if unwrapped is not None:
        cleaned = unwrapped

    if any(marker in cleaned for marker in _PROMPT_LEAK_MARKERS):
        log.warning("review: model echoed prompt instructions instead of answering")
        return "[error] Le modèle n'a pas généré de réponse exploitable. Réessayez."

    if not cleaned:
        log.warning("review: model returned an empty response")
        return "[error] Le modèle n'a pas généré de réponse. Réessayez."

    # Run #b669174a: the answer WAS the file's opening docstring,
    # copied verbatim. Every other guard passed it -- the unwrap check
    # asks whether the content is substantive, and a copy of the input
    # is maximally substantive. See text_cleaning.looks_like_a_copy.
    if file_content and looks_like_a_copy(cleaned, file_content):
        log.warning("review: model returned the file's own content instead of a review")
        return (
            "[error] Le modèle a recopié le fichier au lieu de l'analyser. Réessayez."
        )

    if len(cleaned) > _MAX_REVIEW_OUTPUT_CHARS:
        cleaned = cleaned[:_MAX_REVIEW_OUTPUT_CHARS].rstrip() + "…"
        log.warning("review response truncated to %d chars", _MAX_REVIEW_OUTPUT_CHARS)

    return cleaned


def _read_file_node(state: AgentState) -> AgentState:
    path_str = state.context.get("file_path", state.user_input.strip())
    path = Path(path_str)

    # exists() is inside the try, not before it. Path.exists() returns
    # False for a missing file but RAISES for a malformed one --
    # pathlib ignores ENOENT/ENOTDIR/EBADF/ELOOP and lets everything
    # else through, so ENAMETOOLONG escapes. Seen live on 2026-08-17
    # when the router put an entire pasted document into file_path:
    # this node blew up before reaching any of its own error handling,
    # and the run surfaced to the user as "returned empty output".
    #
    # tools/review.py rejects that shape before dispatch now. This stays
    # anyway: a graph node that raises loses its error message, and the
    # cost of being wrong about which errnos pathlib swallows is paid in
    # unreadable failures.
    try:
        if not path.exists() or not path.is_file():
            state.ok = False
            state.error = f"File not found: {path_str}"
            state.final_output = f"[error] {state.error}"
            return state

        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] cannot read file: {e}"
        return state

    full_chars = len(content)
    if full_chars > _MAX_FILE_CHARS:
        # The marker names Forge on purpose. The old one read
        # "... (truncated at 8000 chars)" in the passive voice, sitting
        # at the end of a file that stops mid-sentence, and on a real
        # run of notes.md the model listed that abrupt ending as defect
        # number one of the document -- "suggests a badly executed
        # copy-paste". It was reviewing our own scissors.
        content = content[:_MAX_FILE_CHARS] + _CUT_MARKER.format(
            shown=_MAX_FILE_CHARS, total=full_chars
        )
        state.context["truncated_from"] = full_chars
        log.warning(
            "review: file truncated to %d chars (of %d)", _MAX_FILE_CHARS, full_chars
        )

    state.context["file_content"] = content
    state.context["file_name"] = path.name
    log.event("review.read_file", name=path.name, chars=len(content))
    return state


def _test_capability_verdict():
    """
    Ask the policy about what this node actually does.

    Checked against tools/test.py's DECLARED requirements, not against
    a registered `test` capability. The distinction matters because
    this graph reaches the module by import: it runs pytest whether or
    not `test` is in ENABLED_TOOLS.

    An earlier version keyed off registration and left a hole, found in
    real use -- with `test` not opted in, POLICY_ALLOW_SUBPROCESS=false
    still spawned a subprocess. The reasoning behind it was that an
    unregistered capability means "no opinion, not a denial", and that
    the gate must never add new reasons for things to stop working.
    The first half was a category error: the flag is a statement about
    what this deployment may do, not about one capability's
    reachability, so a subprocess starting while it is false breaks the
    promise regardless of who started it. The second half confused an
    operator's explicit instruction with an incidental restriction --
    honouring what was asked is not "a new reason".

    REQUIREMENTS is a module constant, so this works with no registry
    and no ENABLED_TOOLS at all.
    """
    from forge.kernel import policy
    from forge.kernel.capability import ToolCapability
    from forge.tools import test as test_tool_module

    return policy.check(
        ToolCapability(
            name="test",
            provider="test",
            handler=test_tool_module.run,
            requirements=test_tool_module.REQUIREMENTS,
            declared=True,
        )
    )


def _run_tests_node(state: AgentState) -> AgentState:
    test_path = state.context.get("test_path")
    if not test_path:
        # Not reached in practice -- build()'s conditional edges skip
        # straight to llm_review when there's no test_path. Kept as a
        # defensive no-op in case this node is ever called directly.
        return state

    # The test step is the `test` capability, so it answers to the
    # policy that governs `test` -- not to review's own profile.
    #
    # This matters in both directions. review declares subprocess=False
    # because reading a file and calling the LLM spawns nothing; that
    # stays true, and review is not denied wholesale on a box where
    # subprocesses are off. But running pytest here IS a subprocess,
    # and it reached one by importing the module rather than by
    # dispatching, so the gate never saw it. Checking here closes that
    # without widening review's own declaration, which would deny the
    # far more common no-tests review for a step it never takes.
    #
    # It also covers POST /run?graph=review, which enters this graph
    # directly and never passes the tool-level check at all.
    verdict = _test_capability_verdict()
    if not verdict:
        log.warning("review: skipping tests, %s", verdict.reason)
        state.context["test_output"] = f"[skipped] {verdict.reason}"
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

    truncated_from = state.context.get("truncated_from")
    truncation_note = (
        _TRUNCATION_NOTE.format(total=truncated_from, shown=_MAX_FILE_CHARS)
        if truncated_from
        else ""
    )

    prompt = _REVIEW_PROMPT.format(
        today_line=today_line(),
        filename=filename,
        question=question,
        content=content,
        test_section=test_section,
        truncation_note=truncation_note,
    )

    # Language named in LAST position, and only when forge.lang is
    # sure -- same treatment recall got in the v3.12 dettes batch, for
    # the same reason: this prompt body is English prose, and it pulls
    # a French answer toward English all on its own. Appended rather
    # than templated in, so "last" cannot drift as the template grows.
    language_line = lang.line_for(question)

    log.event("review.llm_call", filename=filename, prompt_chars=len(prompt))
    try:
        # No grammar, so _grammar_for() supplies the ROUTER's. That is
        # deliberate -- see the header of forge/prose_grammar.py.
        raw = call_llm(prompt + language_line)
        log.event("review.raw_output", raw=raw)
        answer = _clean_review_response(raw, content)
        # The deterministic half. Naming the language in the prompt is
        # still a wording fix, and wording fixes have lost seven times
        # on this codebase. The retry re-sends the same prompt with a
        # different final line, so the KV prefix survives and only the
        # tail is recomputed.
        answer = lang.enforce(
            question,
            answer,
            retry=lambda line: _clean_review_response(call_llm(prompt + line), content),
            enabled=ENFORCE_ANSWER_LANGUAGE,
        )
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    # Logged unconditionally (mirrors orchestrator.py's
    # router.raw_output) -- this call previously had no raw-output
    # visibility at all, which made the JSON-habit bug above
    # impossible to confirm from logs alone the first time it happened.
    # Not on an error. "[error] the model copied the file" followed by
    # "partial review: 40 % of the file" reads as though a review
    # happened and covered 40 % of it. Seen on run #9942466c, where the
    # copy guard fired and the footer went on anyway.
    if truncated_from and not answer.startswith("[error]"):
        answer += _TRUNCATION_FOOTER.format(
            shown=_MAX_FILE_CHARS,
            total=truncated_from,
            percent=round(100 * _MAX_FILE_CHARS / truncated_from),
        )

    state.final_output = answer
    state.final_tool = "review"
    log.event(
        "review.done", chars=len(state.final_output), truncated_from=truncated_from
    )
    return state


def _error_node(state: AgentState) -> AgentState:
    """
    Turn a failed node into a message the user can read.

    It used to only flip state.ok and trust whichever node failed to
    have set final_output. That holds for a node that returns an error
    -- and fails exactly when it is needed, for a node that RAISES,
    which leaves final_output empty. The result was a run reported as
    "tool 'review' returned empty output": the least informative thing
    the system can say, produced by the node whose whole job is to say
    something informative.

    So the output is guaranteed here rather than assumed upstream.
    """
    log.warning("review graph: %s", state.error)
    state.ok = True  # surface as message, not crash
    if not state.final_output:
        state.final_output = f"[error] review failed: {state.error or 'unknown error'}"
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
    truncated = state.context.get("truncated_from")
    subtrace.publish(
        subtrace.from_state(
            state,
            {
                "read_file": lambda: (
                    f"{state.context.get('filename', file_path)} — "
                    + (
                        f"{_MAX_FILE_CHARS} caractères lus sur {truncated}"
                        if truncated
                        else f"{len(state.context.get('file_content', ''))} caractères"
                    )
                ),
                "run_tests": lambda: (
                    f"tests : {test_path}" if test_path else "aucun test demandé"
                ),
                "llm_review": lambda: (
                    f"revue générée ({len(state.final_output or '')} caractères)"
                ),
            },
        )
    )
    return state.final_output or ""
