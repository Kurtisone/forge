"""
Tests for deploy/podman_ro_proxy.py's request filter -- the one piece
that actually matters for security here (everything else is stdlib
HTTP plumbing). Imported by file path since deploy/ is an ops script,
not part of the installable forge package.
"""

import importlib.util
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

_PROXY_PATH = Path(__file__).resolve().parents[1] / "deploy" / "podman_ro_proxy.py"
_spec = importlib.util.spec_from_file_location("podman_ro_proxy", _PROXY_PATH)
podman_ro_proxy = importlib.util.module_from_spec(_spec)
sys.modules["podman_ro_proxy"] = podman_ro_proxy
_spec.loader.exec_module(podman_ro_proxy)

_is_allowed = podman_ro_proxy._is_allowed


def test_allows_get_containers_json():
    assert _is_allowed("GET", "/containers/json")
    assert _is_allowed("GET", "/v4.9.0/libpod/containers/json")


def test_allows_get_container_logs():
    assert _is_allowed("GET", "/containers/abc123/logs")
    assert _is_allowed("GET", "/v4.9.0/libpod/containers/abc123/logs?stdout=true")


def test_allows_ping():
    assert _is_allowed("GET", "/_ping")


def test_allows_versioned_ping_like_real_podman_client():
    """Regression test: the real podman CLI pings via a versioned path
    (e.g. /v5.4.2/libpod/_ping) before its actual request, not the
    bare /_ping this originally only matched -- every real podman
    command failed its own connectivity check with a 403 from this
    proxy until this was fixed. Caught in production on 2026-08-11."""
    assert _is_allowed("GET", "/v5.4.2/libpod/_ping")
    assert _is_allowed("GET", "/v5.4.2/_ping")


def test_rejects_non_get_methods_even_on_allowed_paths():
    """The core guarantee: no verb other than GET reaches the real
    socket, no matter the path -- start/stop/rm/exec/pull all go
    through non-GET verbs."""
    for method in ("POST", "DELETE", "PUT", "PATCH"):
        assert not _is_allowed(method, "/containers/json")
        assert not _is_allowed(method, "/containers/abc123/logs")


def test_rejects_mutation_paths():
    mutation_paths = [
        "/containers/abc123/start",
        "/containers/abc123/stop",
        "/containers/abc123/restart",
        "/containers/abc123/kill",
        "/containers/abc123",  # DELETE-a-container path, even as GET must not match unrelated shapes
        "/containers/create",
        "/containers/abc123/exec",
        "/images/pull",
        "/system/prune",
    ]
    for path in mutation_paths:
        assert not _is_allowed("GET", path), f"unexpectedly allowed: GET {path}"


def test_rejects_path_traversal_style_attempts():
    assert not _is_allowed("GET", "/containers/json/../../start")
    assert not _is_allowed("GET", "/containers/abc/logs/../start")


def test_rejects_follow_in_any_form():
    """A follow request is a GET on an allowed path, so the allowlist
    alone lets it straight through -- and podman then never closes
    the connection, parking a handler thread and an upstream socket
    permanently. Three of those and sysadmin's log collection is dead
    without a single rule being broken (audit M-1).

    Every spelling is refused, including the ones podman itself would
    read as false: guessing which strings someone else's decoder
    calls true is how a filter and its target end up disagreeing."""
    for query in (
        "?follow=true",
        "?follow=1",
        "?follow=TRUE",
        "?follow",
        "?follow=false",
        "?stdout=true&follow=true",
        "?FOLLOW=true",
    ):
        path = f"/v4.9.0/libpod/containers/abc123/logs{query}"
        assert not _is_allowed("GET", path), f"unexpectedly allowed: GET {path}"


def test_still_allows_the_query_parameters_sysadmin_actually_sends():
    """The follow check must not turn into a blanket ban on query
    strings -- `podman logs --tail N` is the whole point of this
    endpoint."""
    assert _is_allowed("GET", "/v4.9.0/libpod/containers/abc123/logs?tail=200")
    assert _is_allowed("GET", "/containers/abc123/logs?stdout=true&stderr=true")


# ─── Forwarding behaviour (real sockets, real HTTP) ─────────────────
# The tests above cover the filter in isolation. These start the proxy
# in front of a fake podman on a real Unix socket, because the M-1
# failures -- serialised handling, an upstream that never answers, an
# unbounded body -- only exist in the plumbing the filter never sees.


class _FakeUpstream:
    """Minimal stand-in for podman.sock. `delay` simulates a slow
    podman, `body` a large one."""

    def __init__(self, tmpdir: Path, body: bytes = b"ok", delay: float = 0.0):
        self.path = str(tmpdir / "upstream.sock")
        self._body = body
        self._delay = delay

        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):
                if outer._delay:
                    time.sleep(outer._delay)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(outer._body)))
                self.end_headers()
                self.wfile.write(outer._body)

            def log_message(self, fmt, *args):
                pass

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                # The timeout test deliberately makes the proxy hang up
                # on this fake podman mid-response; a broken pipe here
                # is the expected outcome, not a failure to report.
                pass

        self._server = Server(self.path, Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


class _ProxyUnderTest:
    def __init__(self, tmpdir: Path, upstream_path: str, **handler_kwargs):
        self.path = str(tmpdir / "proxy.sock")
        handler = podman_ro_proxy.make_handler(upstream_path, **handler_kwargs)
        self._server = podman_ro_proxy._UnixSocketHTTPServer(self.path, handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    def get(self, path: str, timeout: float = 10.0):
        conn = podman_ro_proxy._UnixHTTPConnection(self.path, timeout=timeout)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.read(), dict(resp.getheaders())
        finally:
            conn.close()


def test_forwards_an_allowed_request_unchanged(tmp_path):
    with (
        _FakeUpstream(tmp_path, body=b'[{"Names":["forge"]}]') as upstream,
        _ProxyUnderTest(tmp_path, upstream.path) as proxy,
    ):
        status, body, _ = proxy.get("/v4.9.0/libpod/containers/json")
    assert status == 200
    assert body == b'[{"Names":["forge"]}]'


def test_answers_504_when_podman_does_not_respond_in_time(tmp_path):
    """Without a timeout on the upstream connection the handler blocks
    forever on read() and the thread never comes back. 504 also has to
    be distinguishable from the 403 the filter returns, or a wedged
    podman looks like a policy rejection (audit M-1)."""
    with (
        _FakeUpstream(tmp_path, delay=2.0) as upstream,
        _ProxyUnderTest(tmp_path, upstream.path, timeout=0.2) as proxy,
    ):
        status, body, _ = proxy.get("/containers/json")
    assert status == 504
    assert b"timeout" in body.lower()


def test_caps_an_oversized_response_and_corrects_content_length(tmp_path):
    """`?tail=all` against a large journal is an unbounded read into
    the host proxy's memory. Truncating without rewriting
    Content-Length would be worse than the truncation: the client
    waits on bytes that never arrive."""
    with (
        _FakeUpstream(tmp_path, body=b"x" * 5000) as upstream,
        _ProxyUnderTest(tmp_path, upstream.path, max_bytes=1000) as proxy,
    ):
        status, body, headers = proxy.get("/containers/abc/logs")
    assert status == 200
    assert len(body) == 1000
    assert headers["Content-Length"] == "1000"


def test_a_body_exactly_at_the_cap_is_not_truncated(tmp_path):
    """Off-by-one guard: the read asks for cap+1 bytes precisely so an
    exactly-at-cap body isn't reported as cut short."""
    with (
        _FakeUpstream(tmp_path, body=b"x" * 1000) as upstream,
        _ProxyUnderTest(tmp_path, upstream.path, max_bytes=1000) as proxy,
    ):
        status, body, _ = proxy.get("/containers/abc/logs")
    assert status == 200
    assert len(body) == 1000


def test_slow_requests_do_not_queue_behind_each_other(tmp_path):
    """The single-threaded server served one request start to finish
    before looking at the next, so one slow `podman logs` stalled
    everything -- including the client's own /_ping. Four concurrent
    half-second requests should take about half a second, not two."""
    with (
        _FakeUpstream(tmp_path, delay=0.5) as upstream,
        _ProxyUnderTest(tmp_path, upstream.path) as proxy,
    ):
        results: list[int] = []
        lock = threading.Lock()

        def hit():
            status, _, _ = proxy.get("/containers/json")
            with lock:
                results.append(status)

        threads = [threading.Thread(target=hit) for _ in range(4)]
        start = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.monotonic() - start

    assert results == [200, 200, 200, 200]
    # Serialised would be ~2.0s; concurrent is ~0.5s. The threshold is
    # deliberately loose -- this asserts "not serialised", not a
    # latency budget, so a slow CI runner doesn't turn it red.
    assert elapsed < 1.5, f"requests appear to be serialised: {elapsed:.2f}s for 4"
