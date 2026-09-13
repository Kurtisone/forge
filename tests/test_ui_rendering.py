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
            # Explicit: the assertion below reports deno's own stderr,
            # which says what is wrong with the extracted module. A
            # CalledProcessError would say "exit 1".
            check=False,
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


class TestTheQrCodeThePairCommandDraws:
    """
    `!pair` answers with an inline PNG. Before this renderer knew what
    an image was, that reached the screen as `!QR code (data:image/
    png;base64,` followed by nine hundred characters of base64 -- the
    link rule matched, the bang did not, and the bubble filled with
    the payload. Run rather than read, because that is precisely the
    bug reading the source did not catch.
    """

    _PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="

    def test_an_inline_png_becomes_an_image(self, render):
        html = render(f"![QR code]({self._PNG})")

        assert f'<img src="{self._PNG}"' in html
        assert "base64" not in html.replace(self._PNG, ""), (
            "the payload reached the screen as text as well as as an image"
        )

    def test_the_bang_does_not_survive_as_punctuation(self, render):
        assert "!<" not in render(f"![QR code]({self._PNG})")

    def test_the_real_pair_reply_renders_as_an_image(self, monkeypatch, render):
        """
        The actual output of the command, not a hand-written sample:
        the two halves are only correct together.
        """
        from forge import pairing

        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", "http://10.8.0.1:8000")
        monkeypatch.setattr(pairing, "API_TOKEN", "bearer")
        html = render(pairing.build_reply())

        assert '<img src="data:image/png;base64,' in html
        assert "<strong>Appairage</strong>" in html

    def test_a_remote_image_is_not_fetched(self, render):
        """
        Tool output (web_fetch, research, files:read) reaches this
        renderer, so a remote <img> would be a beacon: rendering the
        answer would tell a hostile page it had been read. Downgraded
        to text, the way an unsafe link scheme already is.
        """
        html = render("![pixel](https://tracker.example/p.png)")

        assert "<img" not in html
        assert "tracker.example" in html, "downgraded, not silently dropped"

    @pytest.mark.parametrize(
        "src",
        [
            "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
            "data:text/html;base64,PGgxPmhpPC9oMT4=",
            "javascript:alert(1)",
        ],
    )
    def test_only_png_data_uris_become_images(self, render, src):
        assert "<img" not in render(f"![x]({src})")

    def test_an_image_url_cannot_break_out_of_the_attribute(self, render):
        html = render('![x](data:image/png;base64,AAA" onerror="alert(1))')

        assert "onerror" not in html or "&quot;" in html
        assert "<img" not in html


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
