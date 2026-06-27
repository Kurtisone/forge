"""
Forge review graph: read a file, send it to the LLM for analysis.

This is the first graph that does something the REPL cannot do in a
single turn: it chains two distinct operations (filesystem read, then
LLM reasoning) into a reproducible, traceable pipeline.

Nodes:
  read_file   — reads the target file, puts its content in
                state.context["file_content"]
  llm_review  — builds a review prompt, calls the LLM, stores the answer

Edges:
  read_file → llm_review   (if ok)
  read_file → error        (if file could not be read)

Usage (CLI):
  forge review path/to/file.py
  forge review path/to/file.py "focus on security"

Usage (Python):
  from forge.graphs.review import run
  print(run("src/forge/main.py", question="What can be improved?"))
"""

from pathlib import Path

from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.router.parser import parse_router_output
from forge.types import AgentState

_MAX_FILE_CHARS = 8_000

_REVIEW_PROMPT = """/no_think
You are a code reviewer. Analyse the file below and provide clear,
actionable feedback. Focus on: correctness, readability, performance,
and security. Write in the same language as the question.

File: {filename}
Question: {question}

--- file content ---
{content}
--- end of file ---

Respond in plain text (no JSON). Be concise and specific.
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
        content = content[:_MAX_FILE_CHARS] + f"\n... (truncated at {_MAX_FILE_CHARS} chars)"
        log.warning("review: file truncated to %d chars", _MAX_FILE_CHARS)

    state.context["file_content"] = content
    state.context["file_name"] = path.name
    log.event("review.read_file", name=path.name, chars=len(content))
    return state


def _llm_review_node(state: AgentState) -> AgentState:
    content = state.context.get("file_content", "")
    filename = state.context.get("file_name", "unknown")
    question = state.context.get("question", "Que peut-on améliorer ?")

    prompt = _REVIEW_PROMPT.format(
        filename=filename,
        question=question,
        content=content,
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
    g.add_node("llm_review", _llm_review_node)
    g.add_node("error", _error_node)
    g.add_edge("read_file", "llm_review", condition=lambda s: s.ok)
    g.add_edge("read_file", "error",      condition=lambda s: not s.ok)
    return g


def run(file_path: str, question: str = "Que peut-on améliorer ?") -> str:
    """Review a file and return the LLM's analysis as a plain string."""
    state = build().run(
        file_path,
        initial_context={"file_path": file_path, "question": question},
    )
    return state.final_output or ""
