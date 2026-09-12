"""
A web answer about something that is running on this machine.

Both router prompts already carry the boundary, and research's carries
it with this exact example:

    BOUNDARY: this searches the PUBLIC WEB and knows nothing about this
    machine. A question about one of the user's own services or
    containers ("pourquoi searxng a redémarré") is "sysadmin", never
    here, however recent the event sounds.

Measured over every routing in traces.jsonl on 2026-09-12: 18 research
calls, two of them local questions, and one of the two is that
sentence almost verbatim -- `Pourquoi searxng a redémarré ?`, sent to
the web. The other is the same question with a typo. No sysadmin
routing ever went the other way, so the ambiguity is one-directional.

What is under test is deliberately NOT a re-route. `forge` is a
container here and also a French word; hijacking a web question to
read a container's logs would answer something nobody asked. The
answer stands and gains a line naming what is running.
"""

import pytest

import forge.graphs.research as research_mod
from forge import turn
from forge.graphs.research import names_something_local
from forge.graphs.research import run as research_run

CONTAINERS = ["forge", "forge-llm", "forge-embedding", "searxng"]

#: The two real questions, and the sixteen the same trace file shows
#: going to research legitimately.
LOCAL = [
    "Pourquoi searxng a redémarré ?",
    "pourquoi searxn plante ?",
]
WEB = [
    "Tu peux me faire l'actualité du jeu vidéo s'il te plaît",
    "Tu connais le RLM ? Un nouveau système autour de l'IA",
    "Quelles sont les nouveautés de Qwen3.5 ?",
    "Cherche les actualités sur les modèles Qwen",
    "Tu peux me dire si un nouveau jeu vidéo est sorti aujourd'hui ?",
    "Tu as une idée ce que je pourrais en faire de mon Raspberry Pi 2 ?",
    "Tu peux m'aider à me décider sur quels équipements acheter pour un NAS",
    "actualité du jeu vidéo",
    "Regardes sur le web",
    "Tu peux m'en dire quoi ?",
]


@pytest.mark.parametrize("question", LOCAL)
def test_the_two_real_misroutes_are_recognised(question):
    assert names_something_local(question, CONTAINERS) == "searxng"


@pytest.mark.parametrize("question", WEB)
def test_a_real_web_question_is_left_alone(question):
    assert names_something_local(question, CONTAINERS) is None


def test_nothing_is_claimed_when_podman_cannot_be_asked():
    """
    An empty list means "the proxy is down" as often as it means "no
    containers", and the two are indistinguishable from here.
    """
    assert names_something_local("Pourquoi searxng a redémarré ?", []) is None


def test_a_short_word_never_matches():
    """
    `forge` is five letters and a French word; three-letter tokens
    would make every question local.
    """
    assert names_something_local("et la ?", ["for"]) is None


def _answers(monkeypatch, containers, answer="SearXNG est un métamoteur."):
    monkeypatch.setattr(
        research_mod.web_search,
        "search",
        lambda q: [{"title": "t", "url": "https://x", "content": "c"}],
    )
    monkeypatch.setattr(research_mod.web_fetch, "run", lambda url: "contenu")
    monkeypatch.setattr(research_mod, "call_llm", lambda p, grammar=None: answer)
    monkeypatch.setattr(research_mod.sysadmin, "running_containers", lambda: containers)


def test_the_answer_gains_a_line_and_keeps_its_own(monkeypatch):
    _answers(monkeypatch, CONTAINERS)
    turn.set_input("Pourquoi searxng a redémarré ?")

    answer = research_run("searxng redémarrage")

    assert answer.startswith("SearXNG est un métamoteur.")
    assert "conteneur qui tourne sur cette machine" in answer
    turn.clear()


def test_a_web_answer_gets_no_local_note(monkeypatch):
    """
    Named for what it guards, which is this file's subject: no LOCAL
    footer on a question about something that is not running here.

    It asserted whole-string equality until research started appending
    its sources, which made it fail for a reason that has nothing to do
    with what it tests. The answer is still checked to arrive
    unaltered -- startswith, not a substring, so a note pushed in FRONT
    of it would still fail.
    """
    _answers(monkeypatch, CONTAINERS, answer="Qwen3.5 est sorti en août.")
    turn.set_input("Quelles sont les nouveautés de Qwen3.5 ?")

    answer = research_run("Qwen3.5 nouveautés")
    assert answer.startswith("Qwen3.5 est sorti en août.")
    assert "conteneur qui tourne sur cette machine" not in answer
    turn.clear()


def test_the_turn_is_read_and_not_the_routers_restatement(monkeypatch):
    """
    The router's `content` is where the container name went missing in
    the first place -- it rewrites the question before research ever
    sees it.
    """
    _answers(monkeypatch, CONTAINERS)
    turn.set_input("Pourquoi searxng a redémarré ?")

    answer = research_run("redémarrage service web")

    assert "searxng" in answer
    turn.clear()
