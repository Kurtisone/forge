"""
A run must record what answered it, not what was asked for.

LLM_MODEL is a label. Under llama.cpp it is never even sent -- the
server serves whatever GGUF it was launched with -- so a trace that
records the label records an assumption. Every backend already returns
the real name and Forge was dropping it at the provider boundary, which
is the same thing that happened to token counts and the reason
types.Completion carries usage at all.

It matters now rather than in principle: models get swapped, and a
stored answer nobody can attribute is an answer nobody can re-judge
when the next model disagrees with it.
"""

from forge import metrics
from forge.types import Completion, Usage


def _fresh():
    metrics.clear()
    return metrics.start_run()


class TestWhatAnswered:
    def test_the_model_reaches_the_snapshot(self):
        _fresh()
        metrics.record(Usage(), 10, "Qwen3.5-9B.gguf")
        assert metrics.snapshot()["models"] == ["Qwen3.5-9B.gguf"]

    def test_one_run_many_calls_one_name(self):
        """A router decision and a graph synthesis are two calls."""
        _fresh()
        for _ in range(5):
            metrics.record(Usage(), 10, "Qwen3.5-9B.gguf")
        assert metrics.snapshot()["models"] == ["Qwen3.5-9B.gguf"]
        assert metrics.snapshot()["llm_calls"] == 5

    def test_two_models_in_one_run_are_both_kept(self):
        """
        Why this is a list. Nothing guarantees the calls of one run were
        served by the same thing -- a swap mid-run, or an OpenRouter
        request routed elsewhere on a retry -- and a single field would
        report the first or the last and call it the answer.
        """
        _fresh()
        metrics.record(Usage(), 10, "Qwen3.5-9B.gguf")
        metrics.record(Usage(), 10, "LFM2.5-8B-A1B.gguf")
        assert metrics.snapshot()["models"] == [
            "Qwen3.5-9B.gguf",
            "LFM2.5-8B-A1B.gguf",
        ]

    def test_a_backend_that_says_nothing_adds_nothing(self):
        """Never a placeholder: an empty list means nobody said."""
        _fresh()
        metrics.record(Usage(), 10, "")
        assert metrics.snapshot()["models"] == []
        assert metrics.snapshot()["llm_calls"] == 1

    def test_it_is_still_a_no_op_outside_a_run(self):
        metrics.clear()
        metrics.record(Usage(), 10, "Qwen3.5-9B.gguf")
        assert metrics.snapshot() is None

    def test_a_second_run_does_not_inherit_the_first(self):
        _fresh()
        metrics.record(Usage(), 10, "Qwen3.5-9B.gguf")
        _fresh()
        metrics.record(Usage(), 10, "LFM2.5-8B-A1B.gguf")
        assert metrics.snapshot()["models"] == ["LFM2.5-8B-A1B.gguf"]


class TestTheProvidersReadItBack:
    def test_llama_cpp_takes_the_basename(self, monkeypatch):
        """
        The field is a container path. /models/ is this deployment's
        layout, not information about the model.
        """
        import forge.providers.llama_cpp as prov

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "ok", "model": "/models/LFM2.5-8B-A1B-Q4_K_M.gguf"}

        monkeypatch.setattr(prov.requests, "post", lambda *a, **k: R())
        assert prov.call("http://x", "label", "p").model == "LFM2.5-8B-A1B-Q4_K_M.gguf"

    def test_llama_cpp_survives_a_server_that_omits_it(self, monkeypatch):
        import forge.providers.llama_cpp as prov

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "ok"}

        monkeypatch.setattr(prov.requests, "post", lambda *a, **k: R())
        assert prov.call("http://x", "label", "p").model == ""

    def test_the_default_is_empty_not_the_label(self):
        """
        Completion() with nothing said must not invent a name. The
        whole point is that the label and the answer are different
        things.
        """
        assert Completion(text="x").model == ""


class TestItReachesSomewhereAPersonLooks:
    """
    The gap between recording a fact and surfacing it.

    RunMetrics.models went into the trace record, the roadmap said "which
    model answered", and nothing anywhere rendered it. Reported as "I
    can't see the model used for the answers", which was correct: the
    header names the model loaded NOW -- a different question, and one
    a trace from before a swap answers wrongly.

    Source assertions, same precedent and caveat as test_ui_security.py.
    """

    def _source(self):
        from pathlib import Path

        import forge.api as api_mod

        return (Path(api_mod.__file__).parent / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_the_trace_card_reads_the_models_list(self):
        assert "t.llm && t.llm.models" in self._source()

    def test_it_renders_nothing_rather_than_a_guess(self):
        """
        An empty list means no backend said, which is not the same fact
        as "it was the configured one".
        """
        src = self._source()
        assert '${models ? `<div class="trace-model"' in src

    def test_the_header_trims_the_extension(self):
        src = self._source()
        assert "function shortModel" in src
        assert "shortModel(data.model)" in src

    def test_the_recorded_name_keeps_it(self, monkeypatch):
        """
        The trim is display only. What goes in the trace has to be the
        exact string the backend reported -- two deployments serving
        `foo.gguf` and `foo.Q4.gguf` are not the same model, and a
        record that cannot tell them apart is a record that cannot
        attribute anything.
        """
        import forge.providers.llama_cpp as prov

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "ok", "model": "/models/Qwen3.5-9B.Q4_K_M.gguf"}

        monkeypatch.setattr(prov.requests, "post", lambda *a, **k: R())
        assert prov.call("http://x", "label", "p").model.endswith(".gguf")
