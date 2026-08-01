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
# Separate from history on purpose: history must stay an exact mirror
# of what's persisted to memory.json across turns (see orchestrator.py
# _finish()), so that llama-server can reuse the KV cache for that
# whole prefix turn to turn. step_context holds this run's own
# in-progress tool results instead, placed after history, and is never
# persisted -- it exists only for the remaining steps of the current
# run.
_SENTINEL_STEP_CONTEXT = "\x00STEP_CONTEXT_BLOCK\x00"

# One line each, describing exactly what "content" must contain for
# that tool. Keep these in sync with each tool's own docstring --
# they're deliberately duplicated (prompt text vs. implementation
# contract) rather than generated from one another, since the prompt
# wording matters for what a small local model will actually produce,
# and shouldn't shift silently if a docstring is reworded.
TOOL_DESCRIPTIONS = {
    "chat": (
        "content is your ACTUAL ANSWER to the user, written naturally, in "
        "the same language the user wrote in. Never repeat or rephrase the "
        "user's message as the answer."
    ),
    "code": "content is the code itself, nothing else.",
    "files": (
        "content is a JSON string describing ONE file operation: "
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
        "content is one git subcommand: status, diff, log, show, branch, "
        'or stash, optionally with flags, e.g. "log --oneline -5". '
        "Read-only. Only use this when the user explicitly asks about git "
        "history, status, or diffs."
    ),
    "memory": (
        "content is a JSON string describing ONE memory operation: "
        '{"action":"remember","kind":"decision" or "todo" or "fact","content":"...","project":"..."} '
        'to store something, or {"action":"recall","query":"..."} to search '
        'past entries by meaning. Use kind="decision" for a choice that was '
        'made, kind="todo" for something still to do, kind="fact" for a '
        "plain piece of information worth keeping (equipment, setup, "
        "preferences, anything the user states about themselves or their "
        'environment) -- when unsure, use "fact". "project" is optional. '
        "Only use this tool when the user explicitly asks you to "
        "remember/save something or to recall/look up something they said "
        "before -- never as a side effect of an unrelated answer. "
        'For "recall", ALWAYS also add "done": false to this JSON -- the '
        "raw search results are not a real answer, you need one more step "
        'to phrase them as a natural reply to the user. For "remember", '
        '"done": false is optional; a short confirmation is usually enough.'
    ),
}

# One worked example per tool, shown only if that tool is enabled.
# Keep to one each -- these cost prompt tokens on every single call.
_TOOL_EXAMPLES = {
    "chat": [
        ("Hello", '{"tool":"chat","content":"Hello! How can I help you?"}'),
        (
            "Connais-tu d'autres langages que Python ?",
            (
                '{"tool":"chat","content":"Oui, je connais aussi JavaScript, '
                'C, C++, Rust, Go, entre autres."}'
            ),
        ),
    ],
    "code": [
        (
            "Write Hello World in Python",
            '{"tool":"code","content":"print(\'Hello World\')"}',
        ),
    ],
    "files": [
        (
            "Read src/forge/main.py",
            '{"tool":"files","content":"{\\"action\\":\\"read\\",\\"path\\":\\"src/forge/main.py\\"}"}',
        ),
        # Deliberate second example: without one, the model only ever
        # sees "read" and defaults to answering with code as plain
        # chat text instead of actually persisting it -- observed in
        # real usage ("create a hello world file" produced a code
        # block, never a file, and a later request to edit "the file
        # you created" failed with file-not-found because nothing had
        # ever been written). A small local model needs to see the
        # write shape, not just read about it in the description.
        (
            "Crée un fichier hello.py qui affiche Hello World",
            (
                '{"tool":"files","content":"{\\"action\\":\\"write\\",'
                '\\"path\\":\\"hello.py\\",\\"content\\":\\"print(\'Hello, World!\')\\"}"}'
            ),
        ),
        # Third example: editing an EXISTING named file needs its
        # current content first, or the model just regenerates
        # something from memory/guesswork instead of the real file
        # (observed live: several "remplace X par Y dans hello.go"
        # requests in a row all answered with the ORIGINAL unmodified
        # content, never touching the actual file). "done": false
        # chains into a write on the next step -- see
        # _format_step_context's steering hint right after this,
        # which is what pushes that second step toward "action":
        # "write" instead of just answering in chat.
        (
            "Dans hello.go, remplace Hello World par Bienvenue",
            (
                '{"tool":"files","content":"{\\"action\\":\\"read\\",'
                '\\"path\\":\\"hello.go\\"}","done":false}'
            ),
        ),
    ],
    "shell": [
        ("List the files here", '{"tool":"shell","content":"ls -la"}'),
    ],
    "git": [
        ("What changed in the last commit?", '{"tool":"git","content":"show"}'),
    ],
    "memory": [
        (
            "Remember: we decided to use SQLite-vec for the RAG",
            (
                '{"tool":"memory","content":"{\\"action\\":\\"remember\\",'
                '\\"kind\\":\\"decision\\",\\"content\\":\\"Use SQLite-vec for the RAG\\"}"}'
            ),
        ),
        # Deliberate second example (every other tool gets exactly one):
        # a plain personal fact, not a decision or a todo, is the case
        # that actually failed in real usage before "fact" existed as a
        # kind -- the model needs to see it, not just be told about it.
        (
            "Mémorise, je possède un Steam Deck",
            (
                '{"tool":"memory","content":"{\\"action\\":\\"remember\\",'
                '\\"kind\\":\\"fact\\",\\"content\\":\\"Possède un Steam Deck\\"}"}'
            ),
        ),
        # Third example, same reasoning: recall's raw output is a bullet
        # list, not a sentence -- "done": false is what turns it into a
        # real answer on the next step, and a small model needs to see
        # the field used, not just read about it in prose.
        (
            "Tu peux me lister mon matériel ?",
            (
                '{"tool":"memory","content":"{\\"action\\":\\"recall\\",'
                '\\"query\\":\\"matériel équipement\\"}","done":false}'
            ),
        ),
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
        + _SENTINEL_STEP_CONTEXT
        + "\nDo not continue the conversation above as plain text. Respond to "
        "the new message below with a single JSON object, exactly like the "
        "examples earlier.\n" + "\nUser: " + _SENTINEL_INPUT + "\n"
    )


_MAX_HISTORY_ENTRY = 120  # chars per entry displayed in the prompt
# A files:read result in step_context is about to be reproduced in
# full (with one part changed) on the very next step -- 120 chars
# would guarantee a truncated/hallucinated rewrite for anything past a
# trivial file. Bounded higher instead of left unbounded, since the
# files tool's own read cap (_MAX_READ_BYTES, 64KB) would still be far
# too much for an 8k-token local model's prompt budget alongside
# everything else in it.
_MAX_STEP_CONTEXT_FILE_ENTRY = 4000


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
    #
    # This must stay an exact function of memory.json's persisted
    # history and nothing else -- no per-run tool-result content mixed
    # in (see step_context / _format_step_context below) -- so that
    # this whole block is byte-identical between the last call of one
    # turn and the first call of the next, letting llama-server reuse
    # the KV cache for it instead of invalidating it every turn.
    lines = ["\nContext from earlier in this conversation (for reference only):"]
    for turn in history:
        speaker = "they said" if turn.get("role") == "user" else "you answered"
        content = turn.get("content", "")
        if len(content) > _MAX_HISTORY_ENTRY:
            content = content[:_MAX_HISTORY_ENTRY] + "…"
        lines.append(f"- {speaker}: {content}")

    return "\n".join(lines) + "\n"


def _format_step_context(step_context: list[dict] | None) -> str:
    if not step_context:
        return ""

    # Tool results from earlier steps of THIS run only -- never
    # persisted, never part of `history`. Kept separate specifically
    # so `history` (above) stays a stable, cacheable prefix; this
    # block is the "new" tail that's expected to change every step and
    # isn't meant to be cache-reused across turns.
    lines = ["\nResult from a tool you already called earlier in this turn:"]
    last_was_memory_result = False
    last_was_files_read = False
    for turn in step_context:
        content = turn.get("content", "")
        last_was_memory_result = content.startswith("[memory]")
        # A files read: the "[files] " prefix, but not a write
        # confirmation/error, which both start with "[ok]"/"[error]"
        # right after that prefix.
        inner = content[len("[files] ") :] if content.startswith("[files] ") else ""
        last_was_files_read = bool(inner) and not inner.startswith(("[ok]", "[error]"))
        # A files read specifically needs a much higher cap than other
        # tool results: it's about to be asked to reproduce this
        # content in full with one part changed, on the very next
        # step, and _MAX_HISTORY_ENTRY (120 chars) is sized for
        # compact history summaries, not for something the model must
        # accurately rewrite. Still bounded, just far less aggressively.
        cap = (
            _MAX_STEP_CONTEXT_FILE_ENTRY if last_was_files_read else _MAX_HISTORY_ENTRY
        )
        if len(content) > cap:
            content = content[:cap] + "…"
        lines.append(f"- {content}")

    # Steering hint, added only right after a memory-tool result: in
    # practice a small local model asked to route again after seeing
    # its own tool output tends to just call the same tool again
    # instead of answering with it -- observed as a real loop-guard
    # failure in testing, not a hypothetical. This is a best-effort
    # nudge, not a guarantee; the loop guard in orchestrator.py is the
    # actual safety net if the model still repeats the call.
    #
    # The concrete before/after example below was added after a
    # second round of live testing: the abstract instruction alone
    # ("in your own words") got the model to correctly switch to
    # "tool":"chat" and stop calling memory again, but it then just
    # copied the raw "- [kind] ..." bullet verbatim as its "answer"
    # instead of actually rephrasing it. A small local model follows a
    # worked example far more reliably than a stated rule.
    if last_was_memory_result:
        lines.append(
            'The last "[memory]" line above already contains the answer. '
            'Respond now with "tool":"chat" and write ONE natural sentence '
            "answering the user -- Do NOT call the memory tool again, and "
            'do NOT copy the "- [kind] ..." bullet format verbatim as your '
            'answer. Example: if that line says "- [fact] Possède un Steam '
            'Deck", a good answer is "Tu as un Steam Deck !", not the '
            "bullet line itself."
        )
    elif last_was_files_read:
        # Same reasoning as the memory hint above: a small local model
        # asked to route again right after reading a file tends to
        # either answer with the content as plain chat (never actually
        # modifying the real file) or just call "read" again. This
        # pushes explicitly toward the write step instead, and
        # reminds it to reuse the file it just read as the base for
        # the edit rather than something recalled from memory/guessed.
        lines.append(
            "The file content above is the CURRENT, real content of that "
            'file. Respond now with "tool":"files" and content = '
            '{"action":"write","path":"<same path as above>","content":'
            '"<the FULL file content above, with the requested change '
            'applied>"}. Do NOT call "action":"read" again, and do NOT '
            "just answer in chat -- the user expects the actual file to "
            "be updated, not a description of what to change."
        )

    return "\n".join(lines) + "\n"


def build_router_prompt(
    user_input: str,
    history: list[dict] | None = None,
    step_context: list[dict] | None = None,
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
        template.replace(_SENTINEL_HISTORY, _format_history(history))
        .replace(_SENTINEL_STEP_CONTEXT, _format_step_context(step_context))
        .replace(_SENTINEL_INPUT, user_input)
    )
