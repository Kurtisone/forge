#!/usr/bin/env python3
"""
Read-only reverse proxy in front of podman.sock.

Why this exists: podman.sock exposes podman's FULL REST API --
start/stop/rm/exec/pull/anything. Mounting it "read-only" as a file
(:ro) only stops something from deleting or replacing the socket
*file*; it does nothing to stop a write *request* being sent through
it. There is no "read-only" flag for the socket itself.

So instead of mounting podman.sock into Forge's container at all,
this proxy sits on the HOST, listens on its own Unix socket, and
forwards only two things to the real podman.sock:
  - GET /*/containers/json   (podman ps)
  - GET /*/containers/{id}/logs*  (podman logs)
Everything else -- every other path, every non-GET method -- gets a
403 without ever touching the real socket. THIS proxy's socket is
what gets mounted into Forge's container (via SYSADMIN_PODMAN_URL),
never the real one.

`?follow=true` on the logs path is refused too, even though it's a
GET on an allowed path: it asks podman for a stream that never ends,
which parks a handler thread and an upstream connection for good.
Availability is part of the read-only guarantee -- a proxy that can
be wedged with three allowed requests isn't much of a proxy. Same
reasoning behind the upstream timeout and the response cap below,
and behind serving requests on threads instead of one at a time.

Stdlib only, deliberately -- consistent with the rest of Forge
(web_fetch's HTML parser, the markdown renderer, etc.): one more
moving part with no extra dependency to audit.

Usage:
    python3 podman_ro_proxy.py \\
        --upstream /run/podman/podman.sock \\
        --listen /run/forge-podman-ro-proxy.sock

Then point Forge's container at the listen socket, bind-mounted
read-only, and set in .env.local:
    SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy.sock
"""

from __future__ import annotations

import argparse
import http.client
import os
import re
import socket
import socketserver
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler

# Matches podman's actual REST paths, tolerant of the /vX.Y.Z API
# version prefix podman clients send: e.g. /v4.9.0/libpod/containers/json
_ALLOWED_GET_PATTERNS = [
    re.compile(r"^(/v[\d.]+)?(/libpod)?/containers/json(\?.*)?$"),
    re.compile(r"^(/v[\d.]+)?(/libpod)?/containers/[^/]+/logs(\?.*)?$"),
    # /_ping is harmless and useful for a startup healthcheck of this
    # proxy itself; not a container-data endpoint. Must tolerate the
    # same optional version/libpod prefix as the other two patterns --
    # the real podman CLI pings via e.g. "/v5.4.2/libpod/_ping" before
    # its actual request, not the bare "/_ping" this originally only
    # matched, which made every real podman command fail its own
    # connectivity check with a 403 from this proxy. Caught in
    # production on 2026-08-11.
    re.compile(r"^(/v[\d.]+)?(/libpod)?/_ping$"),
]


def _mentions_follow(path: str) -> bool:
    """True if the query string mentions `follow` at all.

    `GET /containers/{id}/logs?follow=true` never returns: podman
    holds the connection open and keeps writing. This proxy reads the
    whole upstream body before answering, so a single follow request
    parks a handler thread and an upstream connection permanently.
    Repeat it a few times and sysadmin's log collection is dead --
    with no exploit, no mutation, and nothing in the allowlist
    violated.

    The test is presence of the key, not whether its value looks
    true. `?follow`, `?follow=1`, `?follow=TRUE` and `?follow=false`
    are all refused, because deciding which of those podman's own
    decoder reads as true means reimplementing someone else's
    boolean parsing and being right about it -- the classic way a
    filter and the thing it filters end up disagreeing. Forge never
    sends the parameter in any form (graphs/sysadmin.py runs
    `podman logs --tail N`), so refusing all of them costs nothing.
    """
    query = urllib.parse.urlsplit(path).query
    keys = urllib.parse.parse_qs(query, keep_blank_values=True).keys()
    return any(key.lower() == "follow" for key in keys)


def _is_allowed(method: str, path: str) -> bool:
    if method != "GET":
        return False
    if not any(p.match(path) for p in _ALLOWED_GET_PATTERNS):
        return False
    return not _mentions_follow(path)


# Defaults for the two limits below. Both are here because this proxy
# reads the whole upstream response into memory before answering: with
# no timeout a wedged podman parks a handler thread forever, and with
# no cap a `?tail=all` against a large journal is an OOM on the host,
# not in the container.
DEFAULT_UPSTREAM_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client talks TCP by default; podman.sock is a Unix
    socket, so the connect step is overridden to dial that instead."""

    def __init__(self, unix_path: str, timeout: float = DEFAULT_UPSTREAM_TIMEOUT):
        super().__init__("localhost", timeout=timeout)
        self._unix_path = unix_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # HTTPConnection.timeout is honoured for TCP by its own
        # connect(); overriding connect() means applying it here by
        # hand, or the socket inherits the global default (None =
        # block forever) and the timeout argument silently does
        # nothing.
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._unix_path)


def make_handler(
    upstream_path: str,
    timeout: float = DEFAULT_UPSTREAM_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
):
    class Handler(BaseHTTPRequestHandler):
        def _forward(self):
            if not _is_allowed(self.command, self.path):
                self.send_response(403)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(
                    b"forbidden: podman_ro_proxy only allows GET on "
                    b"/containers/json and /containers/{id}/logs, "
                    b"and never with follow\n"
                )
                return

            conn = _UnixHTTPConnection(upstream_path, timeout=timeout)
            try:
                conn.request(self.command, self.path)
                resp = conn.getresponse()
                # Read one byte past the cap so an exactly-at-cap body
                # isn't mislabelled as truncated.
                body = resp.read(max_bytes + 1)
                truncated = len(body) > max_bytes
                if truncated:
                    body = body[:max_bytes]
                    print(
                        f"podman_ro_proxy: truncated response for {self.path} "
                        f"at {max_bytes} bytes",
                        file=sys.stderr,
                    )
                self.send_response(resp.status)
                for header, value in resp.getheaders():
                    lowered = header.lower()
                    if lowered == "transfer-encoding":
                        continue  # avoid double-chunking; we already read the full body
                    if lowered == "content-length" and truncated:
                        # Forwarding upstream's length after cutting the
                        # body would leave the client waiting on bytes
                        # that never arrive -- a hang instead of a short
                        # read, which is worse than the truncation.
                        continue
                    self.send_header(header, value)
                if truncated:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except TimeoutError:
                # Distinct from 403: the request was allowed, podman
                # just didn't answer in time. Saying so keeps a wedged
                # upstream from looking like a policy rejection.
                self.send_response(504)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"gateway timeout: podman did not respond in time\n")
            except OSError as exc:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"bad gateway: {exc}\n".encode())
            finally:
                conn.close()

        def do_GET(self):
            self._forward()

        def do_POST(self):
            self._forward()  # rejected by _is_allowed -- kept explicit, not silently dropped

        def do_DELETE(self):
            self._forward()

        def log_message(self, fmt, *args):  # quieter default logging
            pass

    return Handler


class _UnixSocketHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Threaded on purpose (audit M-1).

    UnixStreamServer alone handles one request at a time, start to
    finish. sysadmin's collect step is a blocking `podman logs` that
    can legitimately take seconds on a busy container, and while it
    runs nothing else -- not even the client's own /_ping -- gets
    served. One slow request became a queue for every request.

    daemon_threads means a hung handler can't keep the process alive
    at shutdown; combined with the upstream timeout above, a stuck
    podman costs one thread for at most that long rather than
    permanently.
    """

    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        """A client that hangs up mid-response is normal (podman's own
        CLI does it on ^C) and shouldn't print a full traceback per
        occurrence -- that's how the operator learns to ignore this
        proxy's output, which is where real errors go to hide."""
        exc = sys.exception()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream", required=True, help="path to the real podman.sock"
    )
    parser.add_argument(
        "--listen", required=True, help="path for this proxy's own socket"
    )
    parser.add_argument(
        "--upstream-timeout",
        type=float,
        default=DEFAULT_UPSTREAM_TIMEOUT,
        help="seconds to wait on podman before answering 504 (default: %(default)s)",
    )
    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=DEFAULT_MAX_RESPONSE_BYTES,
        help="cap on a forwarded response body (default: %(default)s)",
    )
    args = parser.parse_args()

    if os.path.exists(args.listen):
        os.remove(args.listen)

    server = _UnixSocketHTTPServer(
        args.listen,
        make_handler(
            args.upstream,
            timeout=args.upstream_timeout,
            max_bytes=args.max_response_bytes,
        ),
    )
    os.chmod(args.listen, 0o660)  # group-readable only, not world
    print(
        f"podman_ro_proxy: {args.listen} -> {args.upstream} (GET-only, containers/json + logs)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if os.path.exists(args.listen):
            os.remove(args.listen)


if __name__ == "__main__":
    main()
