"""
Tests for deploy/podman_ro_proxy.py's request filter -- the one piece
that actually matters for security here (everything else is stdlib
HTTP plumbing). Imported by file path since deploy/ is an ops script,
not part of the installable forge package.
"""

import importlib.util
import sys
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
