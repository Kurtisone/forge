"""
The router prompt lives here and ONLY here.

If you ever need to tweak how the router is instructed, this is the
single file to touch -- nothing else in the codebase builds or
concatenates prompt text.

The prompt is generated dynamically from the currently enabled tools
(forge.tools.registry.available_tools()), not a fixed chat/code pair.
A tool the operator hasn't opted into via ENABLED_TOOLS never appears
in the prompt, so the router can't be steered toward offering it --
the only way a tool becomes routable is the same ENABLED_TOOLS opt-in
that already gates it for the Graph engine.
"""

_SENTINEL_INPUT = "\x00USER_INPUT\x00"
_SENTINEL_HISTORY = "\x00HISTORY_BLOCK\x00"

# One line each, describing exactly what "content" must contain for
# that tool. Keep these in sync with each tool's own docstring --
# they're deliberately duplicated (prompt text vs. implementation
# contract) rather than generated from one another, since the prompt
# wording matters for what a small local model will actually produce,
# and shouldn't shift silently if a docstring is reworded.
TOOL_DESCRIPTIONS = {
    "chat": (
        'content is your ACTUAL ANSWER to the user, written naturally, in '
        "the same language the user wrote in. Never repeat or rephrase the "
        "user's message as the answer."
    ),
    "code": "content is the code itself, nothing else.",
    "files": (
        'content is a JSON string describing ONE file operation: '
        '{"action":"read","path":"..."} or '
        '{"action":"write","path":"...","content":"..."} or '
        '{"action":"list","path":"..."}. Only use this when the user '
        "explicitly asks to read, write, or list a file."
    ),
    "shell": (
        'content is a single shell command, e.g. "ls -la" or '
        '"python3 script.py". Only use this when the user explicitly asks '
        "to run a command; only allowlisted commands will actually execute."
    ),
    "git": (
        'content is one git subcommand: status, diff, log, show, branch, '
        'or stash, optionally with flags, e.g. "log --oneline -5". '
        "Read-only. Only use this when the user explicitly asks about git "
        "history, status, or diffs."
    ),
}

# One worked example per tool, shown only if that tool is enabled.
# Keep to one each -- these cost prompt tokens on every single call.
_TOOL_EXAMPLES = {
    "chat": [
        ("Hello", '{"tool":"chat","content":"Hello! How can I help you?"}'),
        (
            "Connais-tu d'autres langages que Python ?",
            '{"tool":"chat","content":"Oui, je connais aussi JavaScript, '
            'C, C++, Rust, Go, entre autres."}',
        ),
    ],
    "code": [
        ("Write Hello World in Python", '{"tool":"code","content":"print(\'Hello World\')"}'),
    ],
    "files": [
        (
            "Read src/forge/main.py",
            '{"tool":"files","content":"{\\"action\\":\\"read\\",\\"path\\":\\"src/forge/main.py\\"}"}',
        ),
    ],
    "shell": [
        ("List the files here", '{"tool":"shell","content":"ls -la"}'),
    ],
    "git": [
        ("What changed in the last commit?", '{"tool":"git","content":"show"}'),
    ],
}

# If ENABLED_TOOLS ends up empty (misconfiguration), fall back to this
# rather than emitting a prompt with an empty tool list.
_FALLBACK_TOOLS = ["chat", "code"]


def _tool_enum(tools: list[str]) -> str:
    return " or ".join(f'"{t}"' for t in tools)


def _content_meanings(tools: list[str]) -> str:
    lines = []
    for t in tools:
        desc = TOOL_DESCRIPTIONS.get(t, "content is the input this tool expects.")
        lines.append(f'- tool="{t}": {desc}')
    return "\n".join(lines)


def _examples(tools: list[str]) -> str:
    blocks = []
    for t in tools:
        for user_msg, json_out in _TOOL_EXAMPLES.get(t, []):
            blocks.append(f"User: {user_msg}\n{json_out}")
    return "\n\n".join(blocks)


def _build_template(tools: list[str]) -> str:
    return (
        "/no_think\n"
        "You are Forge, a JSON-routing assistant.\n\n"
        "Return ONLY valid JSON. NO EXPLANATION, NO TEXT OUTSIDE THE JSON.\n\n"
        "Format:\n"
        "{\n"
        f'  "tool": {_tool_enum(tools)},\n'
        '  "content": "non-empty string"\n'
        "}\n\n"
        'Optional: add "done": false to this JSON if you need another step\n'
        "after this one to fully answer (rare — only for multi-step tasks).\n"
        'Omit "done" entirely for a normal, single-step answer.\n\n'
        'WHAT "content" MEANS PER TOOL:\n'
        f"{_content_meanings(tools)}\n\n"
        "RULES:\n"
        "- content MUST NEVER be empty\n"
        "- NEVER return empty string\n"
        "- NEVER return null\n"
        "- NEVER return partial JSON\n"
        "- NEVER add text outside the JSON object\n"
        "- Stop generating immediately after the closing brace\n"
        "- Use the conversation history below only as context; do not repeat it\n\n"
        "Examples:\n\n"
        f"{_examples(tools)}\n"
        + _SENTINEL_HISTORY
        + "\nUser: "
        + _SENTINEL_INPUT
        + "\n"
    )


_MAX_HISTORY_ENTRY = 120  # chars per entry displayed in the prompt


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return ""

    # Deliberately NOT formatted as "User: ... / Assistant: ..." --
    # that pattern visually matches the live turn below it, and local
    # models tend to just continue it as plain dialogue instead of
    # emitting JSON. Bullet-point summaries read as context, not as a
    # conversation to continue.
    #
    # Entries are also truncated: a code paste saved before this fix
    # landed would otherwise blow up the prompt with hundreds of lines.
    lines = ["\nContext from earlier in this conversation (for reference only):"]
    for turn in history:
        speaker = "they said" if turn.get("role") == "user" else "you answered"
        content = turn.get("content", "")
        if len(content) > _MAX_HISTORY_ENTRY:
            content = content[:_MAX_HISTORY_ENTRY] + "…"
        lines.append(f"- {speaker}: {content}")
    lines.append(
        "Do not continue the conversation above as plain text. Respond to "
        "the new message below with a single JSON object, exactly like the "
        "examples earlier."
    )
    return "\n".join(lines) + "\n"


def build_router_prompt(
    user_input: str,
    history: list[dict] | None = None,
    available_tools: list[str] | None = None,
) -> str:
    """
    available_tools defaults to whatever's actually enabled+loaded
    (forge.tools.registry.available_tools()) -- pass it explicitly
    only for tests, or callers that need a fixed tool set regardless
    of runtime config.
    """
    if available_tools is None:
        from forge.tools import registry
        available_tools = registry.available_tools() or list(_FALLBACK_TOOLS)

    template = _build_template(available_tools)
    return (
        template
        .replace(_SENTINEL_HISTORY, _format_history(history))
        .replace(_SENTINEL_INPUT, user_input)
    )
