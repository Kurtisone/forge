"""Tests for forge.tools.web_search (SearXNG-backed)."""

from unittest.mock import patch

import forge.config as cfg
import forge.tools.web_search as web_search_mod


def test_web_search_returns_formatted_results(requests_mock):
    requests_mock.get(
        "http://127.0.0.1:8888/search",
        json={
            "results": [
                {
                    "title": "Result One",
                    "url": "https://example.com/one",
                    "content": "First snippet.",
                },
                {
                    "title": "Result Two",
                    "url": "https://example.com/two",
                    "content": "Second snippet.",
                },
            ]
        },
    )
    r = web_search_mod.run("test query")
    assert "Result One" in r
    assert "https://example.com/one" in r
    assert "First snippet." in r
    assert "Result Two" in r


def test_web_search_empty_query():
    r = web_search_mod.run("   ")
    assert "[error]" in r


def test_web_search_no_results(requests_mock):
    requests_mock.get("http://127.0.0.1:8888/search", json={"results": []})
    r = web_search_mod.run("obscure query")
    assert "[no results]" in r


def test_web_search_respects_max_results(requests_mock, monkeypatch):
    monkeypatch.setattr(cfg, "SEARXNG_MAX_RESULTS", 2)
    monkeypatch.setattr(web_search_mod, "SEARXNG_MAX_RESULTS", 2)
    requests_mock.get(
        "http://127.0.0.1:8888/search",
        json={
            "results": [
                {"title": f"R{i}", "url": f"https://x.com/{i}", "content": "c"}
                for i in range(5)
            ]
        },
    )
    r = web_search_mod.run("query")
    assert "R0" in r
    assert "R1" in r
    assert "R2" not in r


def test_web_search_truncates_long_snippet(requests_mock):
    requests_mock.get(
        "http://127.0.0.1:8888/search",
        json={
            "results": [
                {
                    "title": "Long",
                    "url": "https://example.com",
                    "content": "x" * 500,
                }
            ]
        },
    )
    r = web_search_mod.run("query")
    assert "x" * 301 not in r
    assert "…" in r


def test_web_search_non_json_response(requests_mock):
    requests_mock.get(
        "http://127.0.0.1:8888/search",
        text="<html>not json, json format disabled</html>",
        headers={"Content-Type": "text/html"},
    )
    r = web_search_mod.run("query")
    assert "[error]" in r
    assert "json" in r.lower()


def test_web_search_http_error(requests_mock):
    requests_mock.get("http://127.0.0.1:8888/search", status_code=500)
    r = web_search_mod.run("query")
    assert "[error]" in r
    assert "500" in r


def test_web_search_timeout():
    import requests as requests_lib

    with patch.object(
        web_search_mod.requests,
        "get",
        side_effect=requests_lib.exceptions.Timeout,
    ):
        r = web_search_mod.run("query")
    assert "[error]" in r
    assert "timed out" in r
