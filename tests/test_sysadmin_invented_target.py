"""
The five real sysadmin turns of 2026-09-14, replayed.

Read back from /traces on 2026-09-15 -- the first production evidence
of the Harnais since it was wired in, and the only measurement in this
repo that comes from actual user turns rather than a bench.

    user   : Tu peux vérifier pourquoi mon Deck rame ?
    router : {"target_hint": "forge-podman-ro-proxy",
              "question": "pourquoi le système rame-t-il ?"}

Neither of the two "Deck" questions names anything. The router
attached `forge-podman-ro-proxy` to both, and it had a source: a
`shell` turn at 15:19 that day ran
`podman --url unix:///run/forge-podman-ro-proxy/sock ps`, and it was
still in the rolling history. This is h02's family, already written
down -- the model does not guess a name, it copies the most salient
one in front of it.

WHAT IT COST. `_harnais_path` is `SYSADMIN_USE_HARNAIS and not
target_hint`, so a phantom hint switches route A off; then
`target_missed` fires, correctly, and the user gets "[cible
introuvable]" for a question about their machine that the Harnais had
already answered. Three of the five turns ended that way.

WHY IT IS NOT A PROMPT FIX. router/prompt.py already carries the rule
AND this exact sentence as its example ("Mon Steam Deck rame depuis ce
matin" -> no target_hint). It was followed most of the time and
violated in silence the rest, for the thirteenth time in this
codebase. The replacement is arithmetic between two texts.

WHAT STILL REFUSES. The one turn that genuinely named a missing target
-- "Pourquoi forge-podman-ro-proxy ne fonctionne pas ?" -- still stops
at target_missed, and must: answering it from machine-wide facts is
the substitution of run #83fc443e, rebuilt. (That unit is a systemd
--user unit, and deploy/forge-dbus-proxy.sh exposes the SYSTEM bus, so
it is invisible to discovery by construction. A real gap, and a
deployment one.)
"""

from datetime import datetime

import pytest

import forge.graphs.sysadmin as sysadmin_mod
from forge import non_answer, turn
from forge.harnais.collectors import containers as containers_mod
from forge.harnais.collectors import cpu_ram as cpu_ram_mod
from forge.harnais.collectors import logs as logs_mod

MEMINFO = "MemTotal: 15160368 kB\nMemAvailable: 900480 kB\n"
LOADAVG = "7.80 6.12 5.44 1/1594 2759\n"

#: The machine as it was: searxng and forge-llm up, and
#: forge-podman-ro-proxy nowhere, because a --user unit is not on the
#: system bus the proxy exposes.
CONTAINERS = ("forge", "forge-llm", "searxng")


def _ps(names=CONTAINERS) -> str:
    now = int(datetime.now().timestamp())  # noqa: DTZ005
    return "\n".join(f"{n}\t{now - 3600}\tUp 1 hour" for n in names)


def _discover(cmd, timeout):
    if cmd == sysadmin_mod._DISCOVER_UNITS_CMD():
        return '{"type":"a","data":[[["forge.service","","","","","","",0,"","/"]]]}'
    if cmd[0] == "podman" and "ps" in cmd:
        return "\n".join(CONTAINERS)
    return "podman log line"


@pytest.fixture(autouse=True)
def _the_deck(monkeypatch):
    monkeypatch.setattr(sysadmin_mod, "_run_fixed", _discover)
    monkeypatch.setattr(containers_mod, "_run_fixed", lambda cmd, t: _ps())
    monkeypatch.setattr(
        cpu_ram_mod,
        "_read",
        lambda path: MEMINFO if path == cpu_ram_mod._MEMINFO else LOADAVG,
    )
    monkeypatch.setattr(logs_mod, "_run_fixed", lambda cmd, t: "kernel: amdgpu")
    monkeypatch.setattr(sysadmin_mod, "call_llm", lambda p, grammar=None: "Diagnostic.")
    yield
    turn.clear()


def _replay(user: str, hint: str | None, question: str):
    """One production turn, with the router's own payload."""
    turn.set_input(user)
    return sysadmin_mod.build().run(
        hint or "", initial_context={"target_hint": hint, "question": question}
    )


def _nodes(state) -> list[str]:
    return [step.decision_tool for step in state.trace]


#: (user message, target_hint, question, terminal node) -- the four
#: fields are copied from the traces, the fifth is what should happen.
TURNS = [
    pytest.param(
        "Tu peux vérifier pourquoi mon Deck rame ?",
        "forge-podman-ro-proxy",
        "pourquoi le système rame-t-il ?",
        "context_synthesize",
        id="17h41-deck-rame",
    ),
    pytest.param(
        "Pourquoi  forge-podman-ro-proxy  ne fonctionne pas ?",
        "forge-podman-ro-proxy",
        "pourquoi forge-podman-ro-proxy ne fonctionne pas ?",
        "target_missed",
        id="18h40-le-proxy-lui-meme",
    ),
    pytest.param(
        "Tu peux me dire si mon conteneur searxng va bien ?",
        "searxng",
        "mon conteneur searxng va bien ?",
        "context_synthesize",
        id="18h44-searxng",
    ),
    pytest.param(
        "Tu peux me dire si tout va bien sur forge-llm ?",
        "forge-llm",
        "le conteneur forge-llm fonctionne-t-il correctement ?",
        "context_synthesize",
        id="18h51-forge-llm",
    ),
    pytest.param(
        "Aucune erreur sur mon Deck ?",
        "forge-podman-ro-proxy",
        "y a-t-il des erreurs ou des problèmes sur mon Deck ?",
        "context_synthesize",
        id="19h00-aucune-erreur",
    ),
]


@pytest.mark.parametrize(("user", "hint", "question", "expected"), TURNS)
def test_the_production_turns_end_where_they_should(user, hint, question, expected):
    assert _nodes(_replay(user, hint, question))[-1] == expected


def test_the_two_deck_questions_get_an_answer_rather_than_a_refusal():
    """
    The point of the whole change, stated as the user experiences it:
    a question about the machine gets the machine's state, not
    "[cible introuvable]".
    """
    for user, hint, question, _ in (p.values for p in TURNS[:1] + TURNS[4:]):
        state = _replay(user, hint, question)
        assert not non_answer.is_non_answer(state.final_output)
        assert state.context["target_invented"] == hint
        # and route A really ran: the kernel collector is back, which
        # route C drops, so the logs domain is `journalctl -k` rather
        # than one container's output.
        logs = state.context["world"].current_state("logs")
        assert logs, "route A did not observe the logs domain"
        assert all(f.source == "logs" for f in logs)
        assert any("journalctl -k" in f.key for f in logs)


def test_the_named_miss_still_refuses_and_says_which_name():
    state = _replay(*TURNS[1].values[:3])
    assert non_answer.is_non_answer(state.final_output)
    assert state.final_output.startswith(non_answer.TARGET_MISSED_PREFIX)
    assert "forge-podman-ro-proxy" in state.final_output


class TestWhatCountsAsNamingIt:
    """
    The comparison itself. Every uncertainty has to land on the
    refusal, never on a diagnosis of the wrong subject.
    """

    def _asked(self, hint, question, user=""):
        turn.set_input(user)
        state = type("S", (), {"context": {"target_hint": hint, "question": question}})
        return sysadmin_mod._hint_came_from_the_user(state)

    def test_named_in_the_users_own_words(self):
        assert self._asked("searxng", "pourquoi ?", "et searxng alors ?")

    def test_named_only_in_the_routers_restatement(self):
        """
        Enough on its own. The restatement is written by the model
        that would have invented the name -- but in all five real
        turns it was faithful, and reading it can only ever add a
        refusal.
        """
        assert self._asked("searxng", "pourquoi searxng redémarre ?")

    def test_separators_do_not_hide_the_name(self):
        assert self._asked("forge-llm", "pourquoi forge llm rame ?")
        assert self._asked("forge-llm", "pourquoi forge_llm rame ?")

    def test_in_neither_text_is_a_router_artefact(self):
        assert not self._asked(
            "forge-podman-ro-proxy", "pourquoi le système rame-t-il ?"
        )

    def test_no_question_to_judge_against_keeps_the_refusal(self):
        """
        Direct calls, tests, anything not routed. Nothing proves the
        hint was invented, so nothing changes.
        """
        assert self._asked("forge-llm", "")
        assert self._asked("forge-llm", "   ")

    def test_a_stale_turn_cannot_cause_a_fallback_on_its_own(self):
        """
        orchestrator.py sets turn.set_input() per run and nothing calls
        turn.clear(), so a sub-run reached another way reads the
        previous message. Reading the restatement too is what makes
        that harmless.
        """
        assert self._asked(
            "searxng", "pourquoi searxng redémarre ?", user="parle-moi de cuisine"
        )
