"""
Where the answer came from, appended in code.

The synthesis prompt has always said "cite which source a specific
claim comes from only if it matters" -- a rule asked of the model about
a set this module can enumerate exactly, since the URLs it opened are
sitting in state.context. That is the shape of rule this codebase has
replaced thirteen times, and a citation is a worse case than most: a
plausible URL a model produced is indistinguishable from one it read,
and checking costs the reader the trip.

The two lists are two lists on purpose. RESEARCH_FETCH_TOP_N pages are
opened and their text goes into the prompt; every other search result
contributes one snippet. Calling the second kind a source would
overstate what was read, which is the exact overstatement a sources
block exists to prevent.
"""

from forge.graphs.research import _link, _sources_footer


def _r(url, title=""):
    return {"url": url, "title": title, "content": "snippet"}


def _f(url):
    return {"url": url, "content": "page text"}


class TestWhatItNames:
    def test_the_pages_actually_read(self):
        out = _sources_footer([_r("https://a", "Page A")], [_f("https://a")])
        assert "Sources lues" in out
        assert "[Page A](https://a)" in out

    def test_results_never_opened_are_counted_not_listed(self):
        out = _sources_footer(
            [_r("https://a", "A"), _r("https://b", "B"), _r("https://c", "C")],
            [_f("https://a")],
        )
        assert "https://a" in out
        assert "https://b" not in out
        assert "2 autre(s)" in out

    def test_a_failed_fetch_counts_as_a_snippet_not_as_a_source(self):
        """
        It contributed exactly what an unvisited result did. Counting it
        among the pages read is the overstatement this block prevents.
        """
        results = [_r("https://a", "A"), _r("https://b", "B")]
        out = _sources_footer(results, [_f("https://a")])
        assert "https://b" not in out
        assert "1 autre(s)" in out

    def test_when_nothing_could_be_opened_it_says_so(self):
        out = _sources_footer([_r("https://a", "A")], [])
        assert "Aucune page n'a pu être ouverte" in out
        assert "https://a" in out

    def test_no_results_at_all_says_nothing(self):
        """An empty sources block is worse than none."""
        assert _sources_footer([], []) == ""

    def test_a_result_with_no_url_is_not_a_source(self):
        assert _sources_footer([_r("", "A")], []) == ""


class TestTheLinkSurvivesTheRealWeb:
    def test_a_plain_title_and_url(self):
        assert _link("Titre", "https://x") == "[Titre](https://x)"

    def test_a_parenthesis_in_the_url_falls_back_to_the_bare_url(self):
        """
        Wikipedia puts them in paths. The UI's link rule stops at the
        first ')', so a markdown link here renders broken rather than
        failing loudly.
        """
        url = "https://fr.wikipedia.org/wiki/Python_(langage)"
        assert _link("Python", url) == url

    def test_a_bracket_in_the_title_is_removed_not_escaped(self):
        out = _link("Actu [MAJ] du jour", "https://x")
        assert out == "[Actu MAJ du jour](https://x)"

    def test_an_empty_title_falls_back_to_the_url(self):
        assert _link("", "https://x") == "https://x"

    def test_a_very_long_title_is_cut(self):
        out = _link("T" * 300, "https://x")
        assert len(out) < 140 and out.endswith("](https://x)") and "…" in out

    def test_newlines_in_a_title_cannot_break_the_list(self):
        assert _link("Deux\nlignes", "https://x") == "[Deux lignes](https://x)"


class TestItNeverCitesSourcesForANonAnswer:
    """
    _error_node sets ok=True on purpose, to surface a failure as a
    message rather than a crash -- so ok is not the test, the text is.
    A sources block under "[no results]" would be citing sources for a
    sentence that cites nothing.
    """

    def _no_containers(self, monkeypatch):
        from forge.graphs import research

        monkeypatch.setattr(research.sysadmin, "running_containers", list)
        return research

    def test_a_search_that_found_nothing(self, monkeypatch):
        research = self._no_containers(monkeypatch)
        monkeypatch.setattr(research.web_search, "search", lambda q: [])
        out = research.run("quoi de neuf")
        assert "Sources" not in out
        assert "Aucune page" not in out

    def test_a_search_that_failed(self, monkeypatch):
        research = self._no_containers(monkeypatch)

        def boom(q):
            raise research.web_search.SearchError("searxng down")

        monkeypatch.setattr(research.web_search, "search", boom)
        out = research.run("quoi de neuf")
        assert "Sources" not in out

    def test_a_real_answer_gets_its_sources(self, monkeypatch):
        research = self._no_containers(monkeypatch)
        monkeypatch.setattr(
            research.web_search,
            "search",
            lambda q: [
                {"url": "https://a", "title": "Page A", "content": "snip"},
                {"url": "https://b", "title": "Page B", "content": "snip"},
            ],
        )
        monkeypatch.setattr(research.web_fetch, "run", lambda url: f"texte de {url}")
        monkeypatch.setattr(research, "call_llm", lambda p: "La réponse synthétisée.")
        out = research.run("quoi de neuf")
        assert out.startswith("La réponse synthétisée.")
        assert "Sources lues" in out
        assert "[Page A](https://a)" in out
        assert "[Page B](https://b)" in out
