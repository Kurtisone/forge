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

from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.router.parser import parse_router_output
from forge.tools import test as test_tool
from forge.types import AgentState

_MAX_FILE_CHARS = 8_000
_MAX_TEST_OUTPUT_CHARS = 3_000

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
Respond in plain text (no JSON). Be concise and specific.
"""

_TEST_SECTION_TEMPLATE = """
--- test output ({test_path}) ---
{test_output}
--- end of test output ---
"""


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

    # The prompt asks for plain text, but if the model wraps in JSON,
    # parse_router_output extracts the content cleanly.
    decision = parse_router_output(raw)
    state.final_output = decision.content
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
