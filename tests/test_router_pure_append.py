"""
The pure-append invariant of the router prompt.

Prompt N must be a strict, character-for-character prefix of prompt N+1.
That is not a style preference: llama-server can only continue from the
live slot state when the new prompt extends the old one. Any insertion
in the middle forces a rewind to the last recurrent-state checkpoint
before the insertion point and a replay from there, and past a certain
depth no checkpoint is left and the whole prompt is recomputed.

Measured on this model: ~0.30 ms/token for a pure append, ~1.80 ms/token
for an insertion one tenth of the way from the end, ~12 ms/token for a
full recompute. On a ~4200-token router prompt that is the difference
between roughly one second and roughly fifty.

These tests exist because a regression here has no functional symptom at
all. Every prompt stays correct, every answer stays correct, every other
test stays green, and the only evidence is that runs get slower. Without
an explicit invariant, the property would be re-broken by the first
well-meant edit that appends a line to the end of the template.
"""

from forge.router.prompt import build_router_prompt, render_user_turn

TOOLS = ["chat", "code", "files", "memory"]


def _prompt(user_input, history=None, step_context=None):
    return build_router_prompt(
        user_input,
        history=history,
        step_context=step_context,
        available_tools=TOOLS,
    )


def _extend(history, user_msg, assistant_msg):
    return history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def _first_divergence(a, b):
    """Index of the first differing character, or None if a prefixes b."""
    if b.startswith(a):
        return None
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def test_turn_two_strictly_extends_turn_one():
    first = _prompt("Salut, tu peux m'aider ?")
    history = _extend([], "Salut, tu peux m'aider ?", "Bien sûr, avec quoi ?")
    second = _prompt("Crée un fichier notes.py", history=history)

    assert second.startswith(first), (
        "prompt 2 must extend prompt 1 byte for byte; first divergence at "
        f"char {_first_divergence(first, second)}"
    )
    assert len(second) > len(first)


def test_every_turn_of_a_long_conversation_extends_the_previous_one():
    """
    The single-step case, which is the overwhelming majority of runs.
    Ten turns, so that a regression that only shows up once the history
    block is non-trivial still gets caught.
    """
    history = []
    previous = None
    for turn in range(10):
        user_msg = f"question numéro {turn}"
        current = _prompt(user_msg, history=history)
        if previous is not None:
            assert current.startswith(previous), (
                f"turn {turn} is not a pure append over turn {turn - 1}; "
                f"first divergence at char {_first_divergence(previous, current)}"
            )
        previous = current
        history = _extend(history, user_msg, f"réponse numéro {turn}")


def test_history_header_is_present_from_the_very_first_turn():
    """
    A block that first appears on turn 2 is an insertion in front of
    turn 1's text -- one guaranteed cache miss per conversation. The
    header must therefore be emitted with an empty history too.
    """
    empty = _prompt("première question")
    assert "Conversation so far." in empty


def test_live_turn_and_its_history_rendering_are_the_same_bytes():
    message = "Analyse le fichier notes.py et dis-moi ce qui cloche"

    live = _prompt(message)
    later = _prompt("et maintenant ?", history=_extend([], message, "voilà"))

    rendered = render_user_turn(message)
    assert rendered in live
    assert rendered in later
    assert live.count(rendered) == 1
    assert later.count(rendered) == 1


def test_assistant_turns_stay_asymmetric():
    """
    Only the user half is symmetric. Rendering assistant turns as
    "Assistant: ..." would complete a dialogue pattern contradicting what
    the examples teach ("User: X" -> JSON), and nothing requires it --
    an assistant turn never appears as a live line.
    """
    prompt = _prompt("suite", history=_extend([], "salut", "bonjour à toi"))
    assert "(you answered: bonjour à toi)" in prompt
    assert "\nAssistant: bonjour à toi" not in prompt


def test_nothing_is_emitted_after_the_live_user_line():
    """
    The live user line must be the tail of the prompt. Anything after it
    is fixed-length text that every future turn gets inserted in front
    of, which is precisely the layout this branch removed.
    """
    prompt = _prompt("dernière ligne")
    assert prompt.endswith(render_user_turn("dernière ligne"))


def test_a_realistically_long_user_message_is_still_a_pure_append():
    """
    Pins the user cap against a concrete message rather than against the
    constant, which is the only way this catches a cap regression: a test
    written in terms of _MAX_USER_HISTORY_ENTRY passes at any value,
    including the 120 that made every multi-line question diverge.

    Nothing here is unusual -- a pasted traceback, a multi-line question,
    a short function. If a message this size breaks the invariant, the
    invariant does not hold in practice whatever the unit tests say.
    """
    message = (
        "Voici l'erreur que j'obtiens quand je lance les tests dans le "
        "container, et je ne comprends pas d'où elle vient :\n"
        "Traceback (most recent call last):\n"
        '  File "/app/src/forge/router/prompt.py", line 42, in build\n'
        "    return template.replace(sentinel, value)\n"
        "TypeError: replace() argument 2 must be str, not None\n"
        "Est-ce que ça vient de la config ou du template lui-même ? "
        "J'ai vérifié ENABLED_TOOLS et tout a l'air normal de ce côté."
    )
    assert len(message) > 400

    first = _prompt(message)
    second = _prompt("merci", history=_extend([], message, "je regarde ça"))

    assert second.startswith(first), (
        "a message of ordinary length must not diverge; first divergence "
        f"at char {_first_divergence(first, second)}"
    )


def test_a_long_user_message_costs_one_bounded_rewind_not_a_permanent_break():
    """
    A message over _MAX_USER_HISTORY_ENTRY is truncated in history but
    not live, so it does diverge. What matters is that the damage is
    bounded and does not compound: the divergence lands at the cap, not
    at the start of the conversation, and the NEXT turn is a pure append
    again over the now-stable truncated form.
    """
    from forge.router.prompt import _MAX_USER_HISTORY_ENTRY

    huge = "x" * (_MAX_USER_HISTORY_ENTRY + 500)

    first = _prompt(huge)
    history = _extend([], huge, "ok")
    second = _prompt("suite", history=history)

    divergence = _first_divergence(first, second)
    assert divergence is not None  # it really does diverge
    # ...and it lands exactly at the cap, deep inside the last turn --
    # not at the start of it, and not earlier in the prompt.
    assert divergence == first.index(huge) + _MAX_USER_HISTORY_ENTRY

    # And it does not compound: turn 3 extends turn 2 exactly.
    third = _prompt("encore", history=_extend(history, "suite", "d'accord"))
    assert third.startswith(second)


def test_step_context_diverges_only_at_the_tail():
    """
    step_context is this run's own tool output and is never persisted, so
    the first call of the next turn cannot extend the last call of a
    multi-step one. That is accepted rather than fixed.

    What must hold is that the rewind stays a tail. Because step_context
    sits just before the live user line, the rewind covers both -- the
    live line is pushed out of the shared prefix by the block in front of
    it. Everything persisted has to survive, though: if a multi-step run
    invalidated the history block too, one tool call would poison the
    rest of the conversation instead of just its own tail.

    Moving step_context after the live user line would buy back those few
    tokens. Deliberately not done: it would make untrusted tool output
    the last thing the model reads before generating, which is the
    strongest position in the prompt and exactly what the E-2 provenance
    markers exist to defend against.
    """
    history = _extend([], "lis notes.py", "[ok] read 12 bytes")

    last_call_of_turn = _prompt(
        "corrige-le",
        history=history,
        step_context=[{"role": "assistant", "content": "[files] x = 1"}],
    )
    next_turn = _prompt(
        "merci",
        history=_extend(history, "corrige-le", "[ok] written 12 bytes"),
    )

    divergence = _first_divergence(last_call_of_turn, next_turn)
    shared = last_call_of_turn[:divergence]

    assert render_user_turn("lis notes.py") in shared
    assert "(you answered: [ok] read 12 bytes)" in shared

    assert len(last_call_of_turn) - divergence < 2000, (
        "the rewind must stay a tail -- step_context plus the live user "
        "line, not the history block with it"
    )


def test_step_context_grows_by_insertion_close_to_the_end():
    """
    Within a single run, step_context sits between history and the live
    user line, so each step inserts rather than appends. Kept that way on
    purpose: moving it after the user line would make untrusted tool
    output the last thing the model reads before generating, which is the
    strongest position in the prompt and the one the E-2 provenance
    markers exist to defend.

    The trade is only sound while the insertion stays a few tokens from
    the end, well inside checkpoint coverage. This test pins that: the
    divergence must be within the live user line, not further up.
    """
    history = _extend([], "lis notes.py", "[ok] read 12 bytes")
    one_step = _prompt(
        "corrige-le",
        history=history,
        step_context=[{"role": "assistant", "content": "[files] x = 1"}],
    )
    two_steps = _prompt(
        "corrige-le",
        history=history,
        step_context=[
            {"role": "assistant", "content": "[files] x = 1"},
            {"role": "assistant", "content": "[files] written"},
        ],
    )

    divergence = _first_divergence(one_step, two_steps)
    tail_length = len(one_step) - divergence
    assert tail_length < 2000, (
        "step_context must stay near the tail of the prompt; a deep "
        "insertion falls off the checkpoint cliff"
    )


def test_static_prefix_does_not_depend_on_history_or_step_context():
    """
    The static block must be a function of the tool set alone. Anything
    in it that varies per turn invalidates the entire cached prefix, and
    everything downstream of it, on every single call.
    """
    bare = _prompt("q")
    with_everything = _prompt(
        "q",
        history=_extend([], "a", "b"),
        step_context=[{"role": "assistant", "content": "[code] print(1)"}],
    )

    marker = "Conversation so far."
    assert bare.split(marker)[0] == with_everything.split(marker)[0]
