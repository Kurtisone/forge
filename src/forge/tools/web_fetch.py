"""
Sandboxed web-fetch tool.

Fetches a URL over HTTP(S) and returns a truncated, tag-stripped
text version of the response. Two protection layers -- both matter
specifically because Forge's own architecture puts it on a home
network (NiPoGi behind WireGuard, alongside other personal services)
that a router-hallucinated fetch must never be able to reach:

1. SSRF guard (always on, NOT configurable): every IP a hostname
   resolves to is checked against loopback / private / link-local /
   reserved / multicast ranges before any request is made. A router
   decision like "fetch http://192.168.1.1/admin" or
   "http://localhost:8080/api" is rejected before a socket opens.
   Redirects are not followed automatically for the same reason --
   a 200 response from an allowed public host could otherwise
   redirect to an internal address.
2. Domain allowlist (optional, WEB_FETCH_ALLOWED_DOMAINS): empty by
   default, meaning any public domain is fetchable subject to layer
   1. Set it to restrict to a curated list.

Known residual limitation: the IP check and the actual request are
two separate DNS lookups (requests does its own resolution), so a
host that changes its DNS answer between the two (DNS rebinding) is
not fully closed off. Acceptable for a personal single-user tool;
would need a custom transport pinning the resolved IP to close
entirely.

Interface: run(content: str) -> str
  content is a single URL: "https://example.com/page"

To activate: ENABLED_TOOLS=chat,code,web_fetch in .env.local
To restrict: WEB_FETCH_ALLOWED_DOMAINS=wikipedia.org,docs.python.org
"""

import ipaddress
import socket
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import urlparse

import requests

from forge.config import (
    WEB_FETCH_ALLOWED_DOMAINS,
    WEB_FETCH_MAX_BYTES,
    WEB_FETCH_TIMEOUT,
)
from forge.kernel.capability import Requirements
from forge.logger import log

# One outbound HTTP(S) request, behind the SSRF guard.
REQUIREMENTS = Requirements(
    network=True,
    llm=False,
    mutates_workspace=False,
    spawns_process=False,
)


_MAX_OUTPUT_CHARS = 6_000
_REDIRECT_CODES = {301, 302, 303, 307, 308}


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor -- stdlib only, no new dependency."""

    # nav/header/footer/aside are the semantic tags real sites use for
    # menus, sidebars, and language pickers -- observed live on a
    # Wikipedia fetch: without skipping these, the output was almost
    # entirely navigation chrome (179-language list, sidebar menu)
    # with the actual article content pushed past the output cap
    # before it ever appeared.
    _SKIP_TAGS: ClassVar[set[str]] = {
        "script",
        "style",
        "noscript",
        "head",
        "nav",
        "header",
        "footer",
        "aside",
    }

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.chunks.append(stripped)

    def text(self) -> str:
        return "\n".join(self.chunks)


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparsable -- fail closed
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _domain_allowed(hostname: str) -> bool:
    if not WEB_FETCH_ALLOWED_DOMAINS:
        return True
    hostname = hostname.lower()
    return any(
        hostname == d or hostname.endswith(f".{d}") for d in WEB_FETCH_ALLOWED_DOMAINS
    )


def run(content: str) -> str:
    url = content.strip()
    if not url:
        return "[error] empty URL"

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"[error] unsupported scheme: {parsed.scheme!r} (only http/https)"
    if not parsed.hostname:
        return "[error] could not parse hostname from URL"

    if not _domain_allowed(parsed.hostname):
        allowed = ", ".join(sorted(WEB_FETCH_ALLOWED_DOMAINS))
        return (
            f"[error] domain {parsed.hostname!r} is not in the allowlist.\n"
            f"Allowed: {allowed}\n"
            f"Add it to WEB_FETCH_ALLOWED_DOMAINS in .env.local to enable it."
        )

    # A hostname can resolve to several IPs, and DNS is effectively
    # attacker-influenced input from the router's perspective, so every
    # returned address is checked -- not just the first.
    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        return f"[error] could not resolve host: {e}"

    resolved_ips = {info[4][0] for info in addrinfo}
    blocked = [ip for ip in resolved_ips if _is_blocked_ip(ip)]
    if blocked:
        log.warning(
            "web_fetch: blocked SSRF attempt to %r (resolves to %s)",
            parsed.hostname,
            blocked,
        )
        return f"[error] host {parsed.hostname!r} resolves to a disallowed address"

    log.event("web_fetch.run", url=url[:120])

    try:
        resp = requests.get(
            url,
            timeout=WEB_FETCH_TIMEOUT,
            headers={"User-Agent": "Forge/1.0 (+local agent runtime)"},
            stream=True,
            allow_redirects=False,
        )
    except requests.exceptions.Timeout:
        return f"[error] request timed out after {WEB_FETCH_TIMEOUT}s"
    except requests.exceptions.RequestException as e:
        return f"[error] request failed: {e}"

    if resp.status_code in _REDIRECT_CODES:
        location = resp.headers.get("Location", "")
        return (
            f"[redirect] {resp.status_code} -> {location!r}\n"
            f"Re-run web_fetch with this URL explicitly if you want to follow it."
        )

    if resp.status_code != 200:
        return f"[error] HTTP {resp.status_code}"

    content_type = resp.headers.get("Content-Type", "")
    if "text" not in content_type and "json" not in content_type:
        return f"[error] unsupported content-type: {content_type!r}"

    raw = resp.raw.read(WEB_FETCH_MAX_BYTES + 1, decode_content=True)
    truncated_by_size = len(raw) > WEB_FETCH_MAX_BYTES
    raw = raw[:WEB_FETCH_MAX_BYTES]
    text_body = raw.decode(resp.encoding or "utf-8", errors="replace")

    if "html" in content_type:
        extractor = _TextExtractor()
        extractor.feed(text_body)
        text_body = extractor.text()

    if len(text_body) > _MAX_OUTPUT_CHARS:
        text_body = text_body[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
    elif truncated_by_size:
        text_body += "\n... (truncated -- response exceeded WEB_FETCH_MAX_BYTES)"

    log.event("web_fetch.done", url=url[:120], chars=len(text_body))
    return text_body or "[empty response]"
