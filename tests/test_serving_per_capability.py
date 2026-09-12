"""
Which backend answers is a property of the work, not of the process.

FORGE_PROVIDER is one value for the whole process, so a routing
decision, a research synthesis and a compaction summary all go to the
same place whether or not that is a good idea. ARCHITECTURE.md's
Niveau 2 is written against a world where it is not -- the Router asks
for a capability, the Registry lists candidates, the Scheduler chooses
-- and nothing has ever had anything to choose between, because the
answer was a module constant before the question was asked.

This is NOT that Scheduler. Each capability still resolves to exactly
one backend, from configuration, deterministically, so
kernel/registry.candidates() still returns one candidate and
_dispatch's hard stop on an ambiguous capability keeps meaning what it
says. And it carries no cost or quality scores: kernel/capability.py
explains at length why numbers nobody measured would let a Scheduler
look informed while deciding on fiction.
"""

import json

import pytest

import forge.llm as llm_mod
import forge.orchestrator as orch_mod
from forge import serving


@pytest.fixture(autouse=True)
def _no_leak():
    assert serving.current() is None
    yield
    assert serving.current() is None, "a capability name outlived its dispatch"


class TestParsing:
    def test_provider_alone(self):
        t = serving._parse("research=openrouter")
        assert t == {"research": serving.Target("openrouter", None)}

    def test_provider_and_model(self):
        t = serving._parse("research=openrouter:z-ai/glm-4.6")
        assert t["research"] == serving.Target("openrouter", "z-ai/glm-4.6")

    def test_a_model_name_may_contain_colons(self):
        """
        partition, not split: only the FIRST colon separates the
        backend from the model, and tags like ':free' are ordinary.
        """
        t = serving._parse("research=openrouter:z-ai/glm-4.6:free")
        assert t["research"].model == "z-ai/glm-4.6:free"

    def test_several_entries_and_stray_whitespace(self):
        t = serving._parse(" research=openrouter , delegate=ollama:qwen3 ")
        assert set(t) == {"research", "delegate"}
        assert t["delegate"] == serving.Target("ollama", "qwen3")

    def test_empty_means_no_opinion(self):
        assert serving._parse("") == {}
        assert serving._parse("   ,  ") == {}


class TestABadEntryIsDroppedNotRaised:
    """
    Refusing to start would turn a typo in one capability into a dead
    assistant, and nothing depends on this knob: it is empty by
    default. The global backend answers instead, loudly.
    """

    @pytest.mark.parametrize(
        "spec",
        [
            "research",  # no '='
            "=openrouter",  # no capability
            "research=",  # no target
            "research=gpt4all",  # not a backend Forge has
            "research=OpenRouter",  # names are exact
        ],
    )
    def test_it_survives(self, spec):
        assert serving._parse(spec) == {}

    def test_a_good_entry_beside_a_bad_one_is_kept(self):
        t = serving._parse("research=nope,delegate=ollama")
        assert set(t) == {"delegate"}


class TestTheSideChannel:
    def test_nothing_is_set_outside_a_dispatch(self):
        assert serving.current() is None

    def test_the_name_is_readable_inside(self):
        with serving.serving("research"):
            assert serving.current() == "research"

    def test_it_is_reset_even_when_the_tool_raises(self):
        """
        The reason this is a context manager. A tool that raises would
        otherwise leave its name behind, and the run's NEXT routing
        decision -- made after _dispatch returns a ToolResult -- would
        be answered by that tool's backend.
        """
        with pytest.raises(RuntimeError), serving.serving("research"):
            raise RuntimeError("boom")
        assert serving.current() is None

    def test_nesting_restores_the_outer_name(self):
        with serving.serving("research"):
            with serving.serving("web_fetch"):
                assert serving.current() == "web_fetch"
            assert serving.current() == "research"


class TestResolution:
    def test_no_config_means_no_target(self, monkeypatch):
        monkeypatch.setattr(serving, "TARGETS", {})
        assert serving.target_for("research") is None

    def test_an_unlisted_capability_falls_through(self, monkeypatch):
        monkeypatch.setattr(
            serving, "TARGETS", {"research": serving.Target("openrouter")}
        )
        assert serving.target_for("chat") is None

    def test_the_routers_own_call_is_configurable_as_router(self, monkeypatch):
        """
        current() is None outside any dispatch, which is exactly the
        router's own call. Naming it gives it a handle.
        """
        monkeypatch.setattr(serving, "TARGETS", {"router": serving.Target("ollama")})
        assert serving.target_for(None) == serving.Target("ollama")


class TestCallLlmUsesIt:
    def _capture(self, monkeypatch):
        seen = {}

        def fake(url, model, prompt, *a, **k):
            seen["model"] = model
            from forge.types import Completion

            return Completion(text="ok")

        monkeypatch.setattr(llm_mod.llama_cpp, "call", fake)
        monkeypatch.setattr(llm_mod.ollama, "call", fake)
        return seen

    def test_the_default_path_is_untouched(self, monkeypatch):
        seen = self._capture(monkeypatch)
        monkeypatch.setattr(serving, "TARGETS", {})
        monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "llama_cpp")
        monkeypatch.setattr(llm_mod, "LLM_MODEL", "global-label")
        llm_mod.call_llm("p")
        assert seen["model"] == "global-label"

    def test_a_configured_capability_changes_backend_and_model(self, monkeypatch):
        seen = self._capture(monkeypatch)
        monkeypatch.setattr(
            serving, "TARGETS", {"research": serving.Target("ollama", "qwen3")}
        )
        monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "llama_cpp")
        monkeypatch.setattr(llm_mod, "LLM_MODEL", "global-label")
        with serving.serving("research"):
            llm_mod.call_llm("p")
        assert seen["model"] == "qwen3"

    def test_a_backend_without_a_model_keeps_the_global_label(self, monkeypatch):
        """
        LLM_MODEL is never sent to llama.cpp, so a capability that only
        moves backends has no reason to invent one.
        """
        seen = self._capture(monkeypatch)
        monkeypatch.setattr(serving, "TARGETS", {"research": serving.Target("ollama")})
        monkeypatch.setattr(llm_mod, "LLM_MODEL", "global-label")
        with serving.serving("research"):
            llm_mod.call_llm("p")
        assert seen["model"] == "global-label"

    def test_a_grammar_warns_against_the_RESOLVED_backend(self, monkeypatch, caplog):
        """
        The whole point of moving this check. A capability pointed at a
        backend with no GBNF, while the process default HAS one, would
        otherwise be told its grammar was honoured -- and the caller's
        parse is unprotected from that moment.
        """
        self._capture(monkeypatch)
        monkeypatch.setattr(serving, "TARGETS", {"research": serving.Target("ollama")})
        monkeypatch.setattr(llm_mod, "FORGE_PROVIDER", "llama_cpp")
        with serving.serving("research"), caplog.at_level("WARNING"):
            llm_mod.call_llm("p", grammar='root ::= "x"')
        assert "cannot constrain sampling" in caplog.text
        assert "research" in caplog.text


class TestTheDispatchMarksIt:
    def test_a_tool_sees_its_own_name(self, monkeypatch):
        from forge.tools.registry import TOOLS

        seen = {}
        monkeypatch.setitem(
            TOOLS, "chat", lambda c: seen.setdefault("cap", serving.current()) or "ok"
        )
        monkeypatch.setattr(
            orch_mod,
            "call_llm",
            lambda prompt: json.dumps({"tool": "chat", "content": "salut"}),
        )
        orch_mod.Orchestrator(max_steps=1).run("salut")
        assert seen["cap"] == "chat"

    def test_a_tool_that_raises_leaves_nothing_behind(self, monkeypatch):
        from forge.tools.registry import TOOLS

        def boom(_c):
            raise RuntimeError("boom")

        monkeypatch.setitem(TOOLS, "chat", boom)
        monkeypatch.setattr(
            orch_mod,
            "call_llm",
            lambda prompt: json.dumps({"tool": "chat", "content": "salut"}),
        )
        orch_mod.Orchestrator(max_steps=1).run("salut")
        assert serving.current() is None
