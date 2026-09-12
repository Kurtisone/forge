"""
The checks must apply to the path the grammar guarantees.

`parse_router_output` ran its repetition guard on the whole raw output
at step 0 and its leak check at step 4, the plain-text fallback. A
model under the GBNF grammar emits a valid JSON object, so step 1
returned before step 4 was ever reached -- and a repetition loop or a
block of leaked prompt sitting INSIDE `content` went through untouched
and was spoken to the user as the answer.

The grammar was doing exactly its job. Guaranteeing the shape of the
output is what made the output skip the guards.

Measured on 2026-09-12 by swapping the served model: a 9B produced
neither failure across 36 router fixtures, LFM2.5-8B-A1B produced nine
-- four replies that were a router envelope nested inside another, and
five that trailed off into reasoning. None was flagged. A check that
nothing exercises is not known to work.
"""

import json

from forge.router.parser import _is_repetition_loop, parse_router_output

# A marker the prompt really contains. test_parser_leak_markers.py is
# what keeps that true.
LIVE_MARKER = "NEVER add text outside the JSON"


def _as_json(payload):
    return json.dumps({"tool": "chat", "content": payload}, ensure_ascii=False)


def test_leaked_prompt_inside_json_is_not_served_as_an_answer():
    """The bug, in one assertion: same text, two envelopes."""
    leak = f"{LIVE_MARKER}. Stop generating immediately after the closing brace."

    as_text = parse_router_output(leak)
    as_json = parse_router_output(_as_json(leak))

    assert as_text.is_fallback, "the plain-text path already caught this"
    assert as_json.is_fallback, (
        "the same leak inside a valid JSON object reached the user"
    )


def test_a_repetition_loop_inside_json_is_not_served_as_an_answer():
    loop = "Je t'ai aidé avec les " + "mises à jour de " * 20

    assert parse_router_output(_as_json(loop)).is_fallback


def test_a_real_answer_inside_json_still_goes_through():
    """
    The direction that must not break. A false positive here costs the
    user their whole reply, which is worse than the bug being fixed.
    """
    real = (
        "TCP garantit l'ordre des paquets et retransmet ceux qui se "
        "perdent, là où UDP les envoie sans accusé de réception, ce qui "
        "le rend plus rapide mais moins fiable pour un transfert de "
        "fichier."
    )
    decision = parse_router_output(_as_json(real))
    assert not decision.is_fallback
    assert decision.content == real


def test_a_json_payload_tool_still_gets_its_nested_object():
    """A files/review payload is JSON in `content` and is not a leak."""
    raw = json.dumps(
        {"tool": "chat", "content": {"file_path": "src/forge/graph.py"}},
        ensure_ascii=False,
    )
    decision = parse_router_output(raw)
    assert not decision.is_fallback
    assert json.loads(decision.content) == {"file_path": "src/forge/graph.py"}


def test_a_repetitive_tool_payload_still_round_trips():
    """
    The direction the first version of this fix broke, and the reason
    the repetition guard applies to `chat` alone.

    A tool payload is data. A config file of near-identical lines, a
    CSV, a fixture -- these repeat themselves because that is what
    they are, and "the text repeats" is a claim about prose
    degenerating, not about a file being dull. Writing one must not be
    rewritten into an apology.
    """
    body = "".join(f"host{i}.local ALL=(ALL) NOPASSWD: ALL\n" for i in range(50))
    # `code` rather than `files` because the floor of _valid_tools() is
    # {"chat", "code"} and this test must not depend on ENABLED_TOOLS.
    # The principle under test is the same: anything that is not `chat`
    # carries data, and data is allowed to repeat itself.
    raw = json.dumps({"tool": "code", "content": body}, ensure_ascii=False)
    decision = parse_router_output(raw)
    assert not decision.is_fallback
    assert decision.content == body


def test_the_same_repetition_as_prose_is_still_caught():
    """The mirror: identical text, spoken instead of written."""
    prose = "mise à jour de " * 30
    assert parse_router_output(_as_json(prose)).is_fallback


class TestTheRepetitionDetectorSeesPhrases:
    """
    The old detector compared ONE token's share against 0.6. That share
    is bounded by the length of the repeating unit -- 50% for a
    two-word loop, 33% for three, 25% for four -- so every loop longer
    than one word was invisible by arithmetic, not by bad luck.
    """

    def test_the_single_word_loop_still_caught(self):
        assert _is_repetition_loop("Allo " * 30)

    def test_the_two_word_loop(self):
        assert _is_repetition_loop("mises jour " * 30)

    def test_the_three_word_loop(self):
        assert _is_repetition_loop("mises à jour " * 20)

    def test_the_loop_that_was_actually_produced(self):
        assert _is_repetition_loop(
            "Je t'ai aidé avec mon Steam Deck et les dernières "
            + "mises à jour de " * 20
        )

    def test_prose_is_not_a_loop(self):
        assert not _is_repetition_loop(
            "Le protocole TCP garantit l'ordre des paquets et retransmet "
            "ceux qui se perdent, là où UDP se contente de les envoyer "
            "sans accusé de réception, ce qui le rend plus rapide mais "
            "moins fiable pour un transfert de fichier."
        )

    def test_a_list_of_similar_steps_is_not_a_loop(self):
        assert not _is_repetition_loop(
            "- lire le fichier\n- vérifier les tests\n- relancer le "
            "service\n- lire le journal\n- vérifier les tests à nouveau"
        )

    def test_repetitive_code_is_not_a_loop(self):
        assert not _is_repetition_loop(
            "def f(x):\n    return x + 1\n\ndef g(x):\n    return x + 2\n"
            "\ndef h(x):\n    return x + 3"
        )

    def test_too_short_to_judge(self):
        assert not _is_repetition_loop("Allo Allo Allo")


class TestTheTruncatedEnvelope:
    """
    The dominant failure of LFM2.5-8B-A1B in the 2026-09-12 comparison:
    four replies out of thirty-six were a router envelope restated
    inside the content of another one, cut off before it closed. The
    cascade returned the outer object, so the user was shown a raw JSON
    fragment as the answer to their question.

    Three conditions, all required -- opens a brace, names both
    protocol keys, does not parse. Across 72 recorded replies from two
    models, the four that matched are exactly the four that were
    broken, and no real answer matched even one condition.
    """

    def test_the_shape_that_reached_the_user(self):
        broken = '{\n  "tool": "chat",\n  "content": "Sur quel appareil je développe, déjà ?"'
        assert parse_router_output(_as_json(broken)).is_fallback

    def test_a_complete_second_envelope_is_unwrapped_not_discarded(self):
        """
        The model wrapped a good answer one level too deep. The
        cascade cannot see the inner object -- it sits inside a JSON
        string, so its quotes are escaped and _all_json_objects()
        scanning the raw text finds nothing valid there. Only the
        outer parses, and its content is the escaped mess. Unwrapped
        rather than refused: the answer is in there.
        """
        inner = json.dumps({"tool": "chat", "content": "La vraie réponse."})
        decision = parse_router_output(_as_json(inner))
        assert not decision.is_fallback
        assert decision.content == "La vraie réponse."

    def test_envelopes_all_the_way_down_are_refused_not_looped_on(self):
        """Bounded unwrapping: model output does not get a while True."""
        text = "réponse"
        for _ in range(6):
            text = json.dumps({"tool": "chat", "content": text})
        decision = parse_router_output(_as_json(text))
        assert decision.is_fallback

    def test_an_answer_that_merely_mentions_json_is_untouched(self):
        real = (
            'Le routeur répond avec un objet {"tool": ...} mais je ne peux '
            "pas te montrer le mien ici."
        )
        assert not parse_router_output(_as_json(real)).is_fallback

    def test_an_answer_that_is_valid_json_data_is_untouched(self):
        """A chat answer may legitimately BE json -- just not the envelope."""
        data = '{"cpu": "AMD Van Gogh", "ram": "16 Go"}'
        decision = parse_router_output(_as_json(data))
        assert not decision.is_fallback
        assert decision.content == data
