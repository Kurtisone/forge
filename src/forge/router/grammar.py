"""
GBNF grammar for llama.cpp's grammar-constrained decoding
(https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md).

Constrains sampling so the model can ONLY produce tokens matching the
router's exact JSON contract -- {"tool": ..., "content": ..., "done": ...}
-- at the sampling level, instead of catching malformed output after
the fact in router/parser.py's fallback chain (repetition loops,
leaked prompt text, hallucinated new dialogue turns, empty output).
Those fallback paths stay in place as defense-in-depth -- a different
provider, a grammar-disabled setup, or a bug in this grammar itself
still needs them -- but with grammar enabled they should rarely, if
ever, trigger.

Only meaningful for llama.cpp (providers/llama_cpp.py) -- the only
provider Forge talks to that exposes raw GBNF grammar sampling.
Ollama has a coarser "format": "json" (valid JSON, no schema
enforcement); OpenRouter/OpenAI-style APIs have "response_format":
{"type": "json_object"} with the same limitation. Neither can pin
"tool" to one of a specific set of literal values the way GBNF can.
"""

_TOOL_SENTINEL = "%%TOOL_ALTERNATION%%"

# Mirrors router.prompt's own fallback: if ENABLED_TOOLS somehow
# resolves empty, don't hand llama.cpp a grammar with an empty
# alternation (which would make "tool" unsatisfiable).
_FALLBACK_TOOLS = ["chat", "code"]

# A generic "any JSON string" body (schar/hex) plus a "boolean" rule
# for the optional "done" field. Deliberately NOT full JSON (no
# numbers, arrays, nested objects) -- content is always a string in
# every RouterDecision, so the grammar doesn't need to allow anything
# else there.
_TEMPLATE = (
    r'root    ::= "{" ws "\"tool\"" ws ":" ws tool ws "," ws "\"content\"" '
    r'ws ":" ws string (ws "," ws "\"done\"" ws ":" ws boolean)? ws "}"' "\n"
    r"tool    ::= " + _TOOL_SENTINEL + "\n"
    r'string  ::= "\"" schar* "\""' "\n"
    r'schar   ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)' "\n"
    r"hex     ::= [0-9a-fA-F]" "\n"
    r'boolean ::= "true" | "false"' "\n"
    r"ws      ::= [ \t\n]*" "\n"
)


def _tool_alternation(tools: list[str]) -> str:
    # Each alternative is a GBNF string terminal matching the literal
    # quoted tool name, e.g. "\"chat\"" matches the 6-character JSON
    # substring "chat" (with its quotes).
    return " | ".join('"\\"' + t + '\\""' for t in tools)


def build_router_grammar(available_tools: list[str] | None = None) -> str:
    """
    available_tools defaults to whatever's actually enabled+loaded
    (forge.tools.registry.available_tools()), same convention as
    router.prompt.build_router_prompt -- pass it explicitly only for
    tests or callers that need a fixed tool set regardless of runtime
    config.
    """
    if available_tools is None:
        from forge.tools import registry
        available_tools = registry.available_tools() or list(_FALLBACK_TOOLS)

    return _TEMPLATE.replace(_TOOL_SENTINEL, _tool_alternation(available_tools))
