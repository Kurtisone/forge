"""
Tests for the UI's XSS hardening (security audit, C-3).

Two of the three fixes live in JavaScript inside static/index.html,
which this suite has no runtime for -- so those two are covered by
asserting on the source itself. That is a weaker test than executing
the renderer, and deliberately so: it's a regression tripwire, not a
proof. It catches the realistic failure mode (someone simplifies
escapeHtml back to three replaces, or rewrites the link rule as a
plain string interpolation) without pulling a JS engine into the
test dependencies for one file.

The third fix (response headers) is a real behavioural test.
"""

from pathlib import Path

from fastapi.testclient import TestClient

import forge.api as api_mod

_INDEX = Path(api_mod.__file__).parent / "static" / "index.html"


def _index_source() -> str:
    return _INDEX.read_text(encoding="utf-8")


# ── escapeHtml covers attribute context ──────────────────────────────


def test_escape_html_escapes_quotes():
    src = _index_source()
    assert "&quot;" in src, "escapeHtml must escape double quotes"
    assert "&#39;" in src, "escapeHtml must escape single quotes"


# ── link rendering validates the scheme ──────────────────────────────


def test_link_rule_is_not_a_raw_interpolation():
    """
    The vulnerable form was a direct regex replacement string:
        '<a href="$2" target="_blank" rel="noopener">$1</a>'
    which put unvalidated URL text straight into an attribute.
    """
    assert '<a href="$2"' not in _index_source()


def test_link_scheme_allowlist_exists():
    assert "_SAFE_LINK_SCHEME" in _index_source()


# ── response headers ─────────────────────────────────────────────────


def test_ui_sends_content_security_policy():
    r = TestClient(api_mod.app).get("/")
    assert r.status_code == 200
    csp = r.headers.get("content-security-policy")
    assert csp is not None
    # The clause that matters most: an injected script cannot ship the
    # localStorage API token off-host.
    assert "connect-src 'self'" in csp
    assert "default-src 'self'" in csp


def test_ui_sends_nosniff():
    r = TestClient(api_mod.app).get("/")
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_code_fence_language_badge_is_escaped():
    """The fence tag is rendered as a visible badge (the block was
    coloured but nothing said whether it was Go or Python). \\w+ can't
    carry markup today, but the badge must not be the one interpolation
    in this file that trusts its input -- same rule as the link rule
    above."""
    html = _index_source()
    assert 'class="code-lang"' in html
    assert "${escapeHtml(lang)}" in html
    assert "${lang}" not in html


def test_diff_blocks_get_the_badge_too():
    """renderDiff takes the badge rather than building its own, so the
    two block kinds can't drift apart in how they label themselves."""
    html = _index_source()
    assert "function renderDiff(code, badge = '')" in html
    assert "renderDiff(body, badge)" in html
