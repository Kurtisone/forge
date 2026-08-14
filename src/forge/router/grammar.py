"""
GBNF grammar for llama.cpp's grammar-constrained decoding
(https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md).

Constrains sampling so the model can ONLY produce tokens matching the
router's exact JSON contract -- {"tool": ..., "content": ..., "done": ...},
where "content" is a nested JSON object for the tools whose payload
is itself JSON, and a plain string for the rest
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

from forge.tool_payload import JSON_PAYLOAD_TOOLS

# Mirrors router.prompt's own fallback: if ENABLED_TOOLS somehow
# resolves empty, don't hand llama.cpp a grammar with an empty
# alternation (which would make "tool" unsatisfiable).
_FALLBACK_TOOLS = ["chat", "code"]

# The shape of "content" is conditioned on WHICH tool was picked,
# which GBNF can express because "tool" is pinned before "content" in
# every call rule.
#
# For a tool whose payload is itself JSON (JSON_PAYLOAD_TOOLS),
# content MUST be a nested object. Offering "string | object" for
# every tool -- the first attempt at this fix -- did not work: it left
# the escaped-string shape reachable, and against a 9B's prior for it
# the worked examples alone lost. Observed live on the first real
# files:write, which came back as an escaped string and died on
# `import "fmt"`.
#
# That failure is not recoverable after the fact, and this is the
# reason the constraint has to live in the grammar rather than in the
# parser. In the string shape the payload sits inside a JSON string,
# so file content needs DOUBLE escaping (\\n, \\\") -- and the
# grammar cannot check it, because to schar the whole payload is just
# characters. Under-escaped quotes then terminate the string early and
# the result is genuinely ambiguous: no lenient parse recovers it.
# In the object shape the file body is a plain JSON string, so schar
# (which excludes bare " and control characters) enforces its escaping
# at the sampling level. The failure mode stops being caught and
# starts being impossible.
#
# router/parser.py re-encodes an object back into
# RouterDecision.content, so nothing downstream sees a new type, and
# it still ACCEPTS the string shape -- providers without GBNF need it.
#
# Objects pull in full JSON values (arrays, numbers, null): a payload
# field could legitimately hold any of them, and the model shouldn't
# be steered into stringifying a number to satisfy the grammar.
_SHARED_RULES = (
    r'object  ::= "{" ws (member (ws "," ws member)*)? ws "}"'
    "\n"
    r'member  ::= string ws ":" ws value'
    "\n"
    r'array   ::= "[" ws (value (ws "," ws value)*)? ws "]"'
    "\n"
    r'value   ::= string | object | array | number | boolean | "null"'
    "\n"
    r'string  ::= "\"" schar* "\""'
    "\n"
    r'schar   ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)'
    "\n"
    r"hex     ::= [0-9a-fA-F]"
    "\n"
    r'number  ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?'
    "\n"
    r'boolean ::= "true" | "false"'
    "\n"
    r'done    ::= ws "," ws "\"done\"" ws ":" ws boolean'
    "\n"
    r"ws      ::= [ \t\n]*"
    "\n"
)


def _tool_alternation(tools: list[str]) -> str:
    # Each alternative is a GBNF string terminal matching the literal
    # quoted tool name, e.g. "\"chat\"" matches the 6-character JSON
    # substring "chat" (with its quotes).
    return " | ".join('"\\"' + t + '\\""' for t in tools)


def _call_rule(name: str, tool_rule: str, content_rule: str) -> str:
    return (
        f'{name} ::= "{{" ws "\\"tool\\"" ws ":" ws {tool_rule} ws "," ws '
        f'"\\"content\\"" ws ":" ws {content_rule} done? ws "}}"'
    )


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

    payload_tools = [t for t in available_tools if t in JSON_PAYLOAD_TOOLS]
    text_tools = [t for t in available_tools if t not in JSON_PAYLOAD_TOOLS]

    branches: list[str] = []
    rules: list[str] = []
    # A branch is emitted only when it has at least one tool: an empty
    # GBNF alternation is unsatisfiable, and a "files-only" tool set is
    # a perfectly legal ENABLED_TOOLS value.
    if payload_tools:
        branches.append("payload_call")
        rules.append(_call_rule("payload_call", "payload_tool", "object"))
        rules.append(f"payload_tool ::= {_tool_alternation(payload_tools)}")
    if text_tools:
        branches.append("text_call")
        rules.append(_call_rule("text_call", "text_tool", "string"))
        rules.append(f"text_tool ::= {_tool_alternation(text_tools)}")

    root = "root    ::= " + " | ".join(branches)
    return "\n".join([root, *rules]) + "\n" + _SHARED_RULES
