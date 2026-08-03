"""Tests for forge.tools.web_fetch."""

from unittest.mock import patch

import forge.config as cfg
import forge.tools.web_fetch as web_fetch_mod


def _fake_addrinfo(ip: str):
    return [(None, None, None, None, (ip, 0))]


# ── SSRF guard ─────────────────────────────────────────────────────


def test_blocks_loopback():
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("127.0.0.1")):
        r = web_fetch_mod.run("http://localhost:8080/")
    assert "disallowed address" in r


def test_blocks_private_range():
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("192.168.1.1")):
        r = web_fetch_mod.run("http://192.168.1.1/admin")
    assert "disallowed address" in r


def test_blocks_link_local():
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("169.254.169.254")):
        r = web_fetch_mod.run("http://169.254.169.254/latest/meta-data")
    assert "disallowed address" in r


def test_blocks_unresolvable_host():
    import socket as socket_mod

    with patch("socket.getaddrinfo", side_effect=socket_mod.gaierror("nope")):
        r = web_fetch_mod.run("http://nonexistent.invalid/")
    assert "[error] could not resolve host" in r


def test_rejects_non_http_scheme():
    r = web_fetch_mod.run("ftp://example.com/file")
    assert "unsupported scheme" in r


def test_empty_url():
    r = web_fetch_mod.run("")
    assert "[error]" in r


# ── domain allowlist ───────────────────────────────────────────────


def test_domain_allowlist_blocks_others(monkeypatch):
    monkeypatch.setattr(cfg, "WEB_FETCH_ALLOWED_DOMAINS", {"wikipedia.org"})
    monkeypatch.setattr(web_fetch_mod, "WEB_FETCH_ALLOWED_DOMAINS", {"wikipedia.org"})
    r = web_fetch_mod.run("https://example.com/page")
    assert "not in the allowlist" in r


def test_domain_allowlist_allows_subdomain(monkeypatch, requests_mock):
    monkeypatch.setattr(cfg, "WEB_FETCH_ALLOWED_DOMAINS", {"wikipedia.org"})
    monkeypatch.setattr(web_fetch_mod, "WEB_FETCH_ALLOWED_DOMAINS", {"wikipedia.org"})
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")):
        requests_mock.get(
            "https://en.wikipedia.org/wiki/Test",
            text="hello",
            headers={"Content-Type": "text/plain"},
        )
        r = web_fetch_mod.run("https://en.wikipedia.org/wiki/Test")
    assert r == "hello"


# ── happy path / content handling ──────────────────────────────────


def test_fetches_plain_text(requests_mock):
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")):
        requests_mock.get(
            "https://example.com/data",
            text="raw text body",
            headers={"Content-Type": "text/plain"},
        )
        r = web_fetch_mod.run("https://example.com/data")
    assert r == "raw text body"


def test_strips_html_tags(requests_mock):
    html = "<html><head><style>.x{}</style></head><body><h1>Title</h1><p>Hello world</p></body></html>"
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")):
        requests_mock.get(
            "https://example.com/page",
            text=html,
            headers={"Content-Type": "text/html"},
        )
        r = web_fetch_mod.run("https://example.com/page")
    assert "Title" in r
    assert "Hello world" in r
    assert "<" not in r


def test_strips_nav_header_footer_chrome(requests_mock):
    """
    Regression test for a real fetch (Wikipedia): without skipping
    nav/header/footer/aside, a page's navigation chrome (menu,
    language list, sidebar) dominated the output and pushed the
    actual article content past the char cap before it ever
    appeared. script/style/noscript/head alone weren't enough --
    real sites put their non-content chrome in semantic nav/header/
    footer/aside tags.
    """
    html = (
        "<html><body>"
        "<nav>Main page | Contents | Random article | 179 languages</nav>"
        "<header>Site header stuff</header>"
        "<article><h1>Real Title</h1><p>The actual useful content.</p></article>"
        "<aside>Related links sidebar</aside>"
        "<footer>Copyright footer text</footer>"
        "</body></html>"
    )
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")):
        requests_mock.get(
            "https://example.com/article",
            text=html,
            headers={"Content-Type": "text/html"},
        )
        r = web_fetch_mod.run("https://example.com/article")

    assert "Real Title" in r
    assert "The actual useful content" in r
    assert "179 languages" not in r
    assert "Site header stuff" not in r
    assert "Related links sidebar" not in r
    assert "Copyright footer text" not in r


def test_rejects_unsupported_content_type(requests_mock):
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")):
        requests_mock.get(
            "https://example.com/image.png",
            content=b"\x89PNG",
            headers={"Content-Type": "image/png"},
        )
        r = web_fetch_mod.run("https://example.com/image.png")
    assert "unsupported content-type" in r


def test_does_not_follow_redirect_automatically(requests_mock):
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")):
        requests_mock.get(
            "https://example.com/old",
            status_code=302,
            headers={"Location": "http://192.168.1.1/internal"},
        )
        r = web_fetch_mod.run("https://example.com/old")
    assert "[redirect]" in r
    assert "192.168.1.1" in r


def test_http_error_status(requests_mock):
    with patch("socket.getaddrinfo", return_value=_fake_addrinfo("93.184.216.34")):
        requests_mock.get("https://example.com/missing", status_code=404)
        r = web_fetch_mod.run("https://example.com/missing")
    assert "HTTP 404" in r
