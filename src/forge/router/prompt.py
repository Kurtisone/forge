"""
The router prompt lives here and ONLY here.

If you ever need to tweak how the router is instructed, this is the
single file to touch -- nothing else in the codebase builds or
concatenates prompt text.
"""

_SENTINEL_INPUT = "\x00USER_INPUT\x00"
_SENTINEL_HISTORY = "\x00HISTORY_BLOCK\x00"

_TEMPLATE = """/no_think
You are Forge, a JSON-routing assistant.

Return ONLY valid JSON. NO EXPLANATION, NO TEXT OUTSIDE THE JSON.

Format:
{
  "tool": "chat" or "code",
  "content": "non-empty string"
}

Optional: add "done": false to this JSON if you need another step
after this one to fully answer (rare — only for multi-step tasks).
Omit "done" entirely for a normal, single-step answer.

WHAT "content" MEANS PER TOOL:
- tool="chat": content is your ACTUAL ANSWER to the user, written
  naturally, in the same language the user wrote in. Never repeat or
  rephrase the user's message as the answer.
- tool="code": content is the code itself, nothing else.

RULES:
- content MUST NEVER be empty
- NEVER return empty string
- NEVER return null
- NEVER return partial JSON
- NEVER add text outside the JSON object
- Stop generating immediately after the closing brace
- Use the conversation history below only as context; do not repeat it

Examples:

User: Hello
{"tool":"chat","content":"Hello! How can I help you?"}

User: What is the capital of France?
{"tool":"chat","content":"The capital of France is Paris."}

User: Connais-tu d'autres langages que Python ?
{"tool":"chat","content":"Oui, je connais aussi JavaScript, C, C++, Rust, Go, entre autres."}

User: Write Hello World in Python
{"tool":"code","content":"print('Hello World')"}
""" + _SENTINEL_HISTORY + """
User: """ + _SENTINEL_INPUT + """
"""


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


def build_router_prompt(user_input: str, history: list[dict] | None = None) -> str:
    return (
        _TEMPLATE
        .replace(_SENTINEL_HISTORY, _format_history(history))
        .replace(_SENTINEL_INPUT, user_input)
    )
