"""
SearXNG answers 200 with results:[] both when a query matched nothing
and when every engine behind it fell over. Forge used to render both
as "[no results]", so the 2026-08-19 DNS outage -- all five engines in
httpx.ConnectTimeout -- looked exactly like an unlucky query.

These pin the distinction, the shapes the field arrives in, and the
fact that reading it can never break a search.
"""

import pytest
import requests_mock

from forge.tools import web_search

_URL = "http://searxng:8888/search"


def _searxng(m, **body):
    m.get(_URL, json={"results": [], **body})


@pytest.fixture(autouse=True)
def _fixed_url(monkeypatch):
    monkeypatch.setattr(web_search, "SEARXNG_URL", "http://searxng:8888")
    yield


def test_all_engines_down_is_reported_as_a_backend_failure_not_no_results():
    with requests_mock.Mocker() as m:
        _searxng(m, unresponsive_engines=[["google", "timeout"], ["brave", "timeout"]])
        out = web_search.run("actualités")

    assert "[no results]" not in out
    assert "search backend failure" in out
    assert "google" in out and "brave" in out


def test_a_genuinely_empty_result_still_says_no_results():
    with requests_mock.Mocker() as m:
        _searxng(m, unresponsive_engines=[])
        out = web_search.run("zzzzqqq")

    assert "[no results]" in out
    assert "backend failure" not in out


def test_results_win_over_a_partial_outage():
    """One engine down while others answered is a warning, not an
    error: the caller still gets the results it can use."""
    with requests_mock.Mocker() as m:
        m.get(
            _URL,
            json={
                "results": [{"title": "T", "url": "http://x", "content": "c"}],
                "unresponsive_engines": ["brave"],
            },
        )
        out = web_search.run("actualités")

    assert "Search results" in out
    assert web_search.last_unresponsive() == ["brave"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["google"], ["google"]),
        ([["google", "timeout"]], ["google"]),
        ([{"engine": "google", "error": "timeout"}], ["google"]),
        ([{"name": "google"}], ["google"]),
        (None, []),
        ("google", []),
        ([""], []),
    ],
)
def test_every_shape_searxng_has_used_is_read(raw, expected):
    assert web_search._normalize_unresponsive(raw) == expected


def test_an_unparseable_entry_is_still_counted_as_an_outage():
    """An engine name Forge could not read is still evidence the
    backend fell over -- dropping it would hide the outage in order to
    keep the list tidy."""
    assert web_search._normalize_unresponsive([42]) == ["42"]


def test_the_channel_is_cleared_before_each_search():
    with requests_mock.Mocker() as m:
        _searxng(m, unresponsive_engines=["brave"])
        web_search.run("un")
        assert web_search.last_unresponsive() == ["brave"]

    with requests_mock.Mocker() as m:
        _searxng(m, unresponsive_engines=[])
        web_search.run("deux")
        assert web_search.last_unresponsive() == []


def test_a_failed_request_does_not_leave_the_previous_outage_behind():
    with requests_mock.Mocker() as m:
        _searxng(m, unresponsive_engines=["brave"])
        web_search.run("un")

    with requests_mock.Mocker() as m:
        m.get(_URL, status_code=502)
        web_search.run("deux")

    assert web_search.last_unresponsive() == []
