"""
What the web UI actually renders, run rather than read.

test_ui_security.py, test_ui_health_refresh.py and
test_trace_says_when_nothing_answered.py each state the same limit:
no JS runtime, so they assert on the source and are tripwires rather
than proofs. The machine Forge is written on has deno, and the first
time the renderer was executed instead of read it produced a bug that
had been on screen for weeks -- `_emphasis like this_` arriving with
its underscores visible, because inlineMarkdown implements `**bold**`
and `*em*` and nothing else. Both footers research.py appends were
written in the syntax the renderer does not have.

Skipped where deno is absent, which includes CI today. A test that
runs on one machine and skips on another is worth more than no test,
and considerably more than a test that would have to duplicate the
renderer to run everywhere -- a second copy of these functions is a
copy that drifts, and drift in a renderer is invisible until someone
reads their own output and finds punctuation in it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import forge.api as api_mod
from tests.js.extract_renderer import renderer_module

INDEX = Path(api_mod.__file__).parent / "static" / "index.html"

pytestmark = pytest.mark.skipif(
    shutil.which("deno") is None, reason="no deno on this machine"
)


@pytest.fixture(scope="module")
def render(tmp_path_factory):
    """formatContent, as the browser would apply it."""
    work = tmp_path_factory.mktemp("renderer")
    (work / "render.js").write_text(renderer_module(INDEX), encoding="utf-8")

    def _render(text: str) -> str:
        (work / "input.json").write_text(json.dumps(text), encoding="utf-8")
        (work / "run.js").write_text(
            'import { formatContent } from "./render.js";\n'
            'const t = JSON.parse(await Deno.readTextFile("input.json"));\n'
            "console.log(formatContent(t));\n",
            encoding="utf-8",
        )
        out = subprocess.run(
            ["deno", "run", "--quiet", "--allow-read", "run.js"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0, out.stderr
        return out.stdout

    return _render


class TestTheFootersResearchAppends:
    def test_the_sources_header_is_emphasised_not_littered(self, render):
        from forge.graphs.research import _sources_footer

        html = render(
            "Réponse."
            + _sources_footer(
                [{"url": "https://a", "title": "A"}], [{"url": "https://a"}]
            )
        )
        assert "<em>Sources lues :</em>" in html
        assert "_Sources" not in html, "underscore emphasis reaches the screen raw"

    def test_the_sources_are_clickable_and_open_elsewhere(self, render):
        from forge.graphs.research import _sources_footer

        html = render(
            "Réponse."
            + _sources_footer(
                [{"url": "https://a.example/p", "title": "Titre A"}],
                [{"url": "https://a.example/p"}],
            )
        )
        assert 'href="https://a.example/p"' in html
        assert 'target="_blank"' in html and 'rel="noopener noreferrer"' in html
        assert "<ol>" in html, "a numbered source list must render as a list"

    def test_the_local_container_note(self, render):
        from forge.graphs.research import _LOCAL_FOOTER

        html = render("Réponse." + _LOCAL_FOOTER.format(container="forge"))
        assert "<em>À noter :" in html
        assert "_À noter" not in html

    def test_a_bare_url_fallback_is_not_mangled(self, render):
        """
        A URL holding a parenthesis is emitted bare, on purpose. It must
        still survive escaping intact.
        """
        from forge.graphs.research import _link

        url = "https://fr.wikipedia.org/wiki/Python_(langage)"
        html = render("Réponse.\n\n1. " + _link("Python", url))
        assert url in html


class TestTheRulesThisUiDeliberatelyLacks:
    def test_underscores_stay_underscores(self, render):
        """
        Pinned as a DECISION, not an oversight. Adding `_em_` would eat
        the identifiers this assistant's answers are full of.
        """
        html = render("Regarde `file_path` et MAX_STEPS dans RECALL_MAX_DISTANCE.")
        assert "MAX_STEPS" in html and "RECALL_MAX_DISTANCE" in html
        assert "<em>" not in html

    def test_asterisk_emphasis_works(self, render):
        assert "<em>ceci</em>" in render("un *ceci* là")

    def test_bold_works(self, render):
        assert "<strong>ceci</strong>" in render("un **ceci** là")


class TestEscapingStillHoldsWhenExecuted:
    def test_markup_in_an_answer_cannot_inject(self, render):
        html = render('<img src=x onerror="alert(1)">')
        assert "<img" not in html
        assert "&lt;img" in html

    def test_a_javascript_url_never_becomes_a_link(self, render):
        html = render("[clique](javascript:alert(1))")
        assert "<a " not in html
