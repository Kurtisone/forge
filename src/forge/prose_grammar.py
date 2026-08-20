"""
GBNF grammars for calls whose answer is prose, not a routing decision.

Every graph synthesis in Forge asks for plain text and then fights
for it in wording: recall's prompt says "Respond in plain text ONLY.
Do NOT wrap your answer in JSON", names the exact shape to avoid, and
carries a NEVER DO THIS example. review, research and sysadmin do the
same. All four then call call_llm(prompt) with no grammar -- and
providers/llama_cpp._grammar_for() reads a missing grammar as "use
the router's".

So the sampler was admitting *only* {"tool":..., "content":...}. The
instruction was not being ignored; it was unsatisfiable. That is why
recall came back wrapped in router JSON on three runs out of three on
2026-08-19, why try_unwrap_router_json() exists in every graph, and
why the warning "model wrapped a substantive answer in router-style
JSON" has been in the logs for sysadmin and research since v3.11. One
cause, three symptoms, each patched separately at the wording layer.

This is the seventh time on this codebase that a wording fix loses to
a deterministic one. The pattern from router/grammar.py applies
unchanged: a shape you do not want must be made UNREACHABLE during
decoding, not discouraged in the prompt. There, "content ::= string |
object" left the escaped-string branch reachable and the 9B's prior
beat the worked examples; splitting the rule on the tool ended it.

Two shapes, because the callers have two contracts:

  SENTENCE -- one line, no braces anywhere. recall's prompt asks for
    "ONE short, natural sentence"; a memory answer containing a brace
    is not a case worth keeping reachable.

  PROSE -- paragraphs. Only the FIRST character is constrained: it
    cannot be a brace or whitespace, which is enough to make a
    top-level JSON object unreachable, while a brace inside the text
    stays legal. research quotes web pages and sysadmin quotes logs,
    and forbidding braces outright there would trade one wrong output
    for another.

Neither bounds length. Under the router grammar, generation stopped
structurally at the closing brace (16-17 tokens on a routing call);
prose ends when the model emits EOS, which both rules permit at any
point since their tails are starred. n_predict and each graph's own
character cap are what remain. Worth watching in the completion_tokens
of the first real runs.

Only llama.cpp can honour any of this (see call_llm). The
try_unwrap_router_json() calls in the graphs stay exactly where they
are: ollama and openrouter cannot be constrained, and
LLAMA_CPP_USE_GRAMMAR=false has to keep working.

Every construct used here -- negated classes, hex ranges, `*`, `+` --
is already in production in router/grammar.py's _SHARED_RULES. That is
deliberate. There is no GBNF engine in the environment these grammars
are written in, so "valid GBNF" can only be argued, not run; the last
time it was argued, `payload_call` lexed as `payload` plus garbage and
llama-server answered 400 to every completion. Rule names here are
single hyphen-free words for the same reason, and gbnf.validate()
checks them on the way out.
"""

from forge import gbnf

# Not whitespace, not a brace, not a control character. This is the
# whole trick: a router object starts with "{", and a model that
# prepends a space to get around that is blocked too.
_HEAD = r"[^{}\x00-\x20\x7F]"

# Body of a one-liner: no braces, no control characters (so no
# newline either).
_LINE_TAIL = r"[^{}\x00-\x1F\x7F]"

# Body of a paragraph: newline, tab and carriage return allowed,
# other control characters not, braces allowed.
_PROSE_TAIL = r"[^\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"


def _build(tail: str) -> str:
    return f"root ::= head tail*\nhead ::= {_HEAD}\ntail ::= {tail}\n"


SENTENCE = _build(_LINE_TAIL)
PROSE = _build(_PROSE_TAIL)

# Fail at import rather than at the first inference: a grammar
# llama-server cannot parse is a 400 on every completion, and
# _grammar_for() would silently drop it and run unconstrained -- which
# is exactly the state this module exists to leave.
for _g in (SENTENCE, PROSE):
    gbnf.validate(_g)
