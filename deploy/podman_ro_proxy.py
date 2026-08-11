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
from http.server import BaseHTTPRequestHandler

# Matches podman's actual REST paths, tolerant of the /vX.Y.Z API
# version prefix podman clients send: e.g. /v4.9.0/libpod/containers/json
_ALLOWED_GET_PATTERNS = [
    re.compile(r"^(/v[\d.]+)?(/libpod)?/containers/json(\?.*)?$"),
    re.compile(r"^(/v[\d.]+)?(/libpod)?/containers/[^/]+/logs(\?.*)?$"),
    # /_ping is harmless and useful for a startup healthcheck of this
    # proxy itself; not a container-data endpoint.
    re.compile(r"^/_ping$"),
]


def _is_allowed(method: str, path: str) -> bool:
    if method != "GET":
        return False
    return any(p.match(path) for p in _ALLOWED_GET_PATTERNS)


class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client talks TCP by default; podman.sock is a Unix
    socket, so the connect step is overridden to dial that instead."""

    def __init__(self, unix_path: str):
        super().__init__("localhost")
        self._unix_path = unix_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._unix_path)


def make_handler(upstream_path: str):
    class Handler(BaseHTTPRequestHandler):
        def _forward(self):
            if not _is_allowed(self.command, self.path):
                self.send_response(403)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(
                    b"forbidden: podman_ro_proxy only allows GET on "
                    b"/containers/json and /containers/{id}/logs\n"
                )
                return

            conn = _UnixHTTPConnection(upstream_path)
            try:
                conn.request(self.command, self.path)
                resp = conn.getresponse()
                body = resp.read()
                self.send_response(resp.status)
                for header, value in resp.getheaders():
                    if header.lower() == "transfer-encoding":
                        continue  # avoid double-chunking; we already read the full body
                    self.send_header(header, value)
                self.end_headers()
                self.wfile.write(body)
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


class _UnixSocketHTTPServer(socketserver.UnixStreamServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="path to the real podman.sock")
    parser.add_argument("--listen", required=True, help="path for this proxy's own socket")
    args = parser.parse_args()

    if os.path.exists(args.listen):
        os.remove(args.listen)

    server = _UnixSocketHTTPServer(args.listen, make_handler(args.upstream))
    os.chmod(args.listen, 0o660)  # group-readable only, not world
    print(f"podman_ro_proxy: {args.listen} -> {args.upstream} (GET-only, containers/json + logs)")
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
