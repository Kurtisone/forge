"""
Tests that keep the two host proxies mounted the same way.

Symptom, 2026-08-21: `systemctl --user restart forge-podman-ro-proxy`
succeeded, and the container then reported

    dial unix /run/forge-podman-ro-proxy.sock: connect: connection
    refused

which reads as "the proxy is down". The proxy was up. podman_ro_proxy
unlinks and recreates its socket on every start, so the file gets a new
inode; the container had bind-mounted the socket FILE, resolved once at
container start, and kept pointing at the old unlinked one. Recreating
the CONTAINER fixed it. Restarting the proxy never would have.

forge-dbus-proxy never had this, because xdg-dbus-proxy puts its socket
in its own directory and the compose file mounts the DIRECTORY -- names
inside a mounted directory resolve on every access. The two proxies had
different shapes for no reason anyone chose, and only one of them was
survivable.

These tests pin the shape, across the four files that have to agree.
Nothing else compares them: the setup script, the unit, the compose
example and the docs each looked right on their own for months.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

UNIT = ROOT / "deploy" / "systemd" / "forge-podman-ro-proxy.service"
COMPOSE = ROOT / "deploy" / "compose.example.yaml"
SETUP = ROOT / "deploy" / "setup-sysadmin-host-access.sh"
PROXY = ROOT / "deploy" / "podman_ro_proxy.py"


def test_the_podman_socket_sits_inside_its_own_directory():
    """
    A bare `.sock` in $XDG_RUNTIME_DIR is the shape that cannot be
    mounted survivably.
    """
    unit = UNIT.read_text()

    assert "--listen %t/forge-podman-ro-proxy/sock" in unit
    assert "forge-podman-ro-proxy.sock" not in unit


def test_the_unit_does_not_use_runtimedirectory():
    """
    RuntimeDirectory= would create the directory and then DELETE it
    when the unit stops -- reintroducing exactly the disappearing
    inode this layout exists to avoid, one level up.
    """
    directives = [
        line
        for line in UNIT.read_text().splitlines()
        if line.strip().startswith("RuntimeDirectory=")
    ]

    assert not directives, directives


def test_the_proxy_creates_its_own_socket_directory():
    """
    Since the unit cannot own the directory (above), the process has
    to. Otherwise a first run on a clean host fails on a missing path.
    """
    assert "os.makedirs" in PROXY.read_text()


def test_both_proxies_are_mounted_as_directories():
    """
    The claim that actually matters, and the asymmetry that caused the
    bug: whatever the container mounts must be a directory for BOTH,
    so a recreated socket inside it is found.
    """
    compose = COMPOSE.read_text()

    for mount in (
        "${XDG_RUNTIME_DIR}/forge-dbus-proxy:/run/forge-dbus-proxy:ro",
        "${XDG_RUNTIME_DIR}/forge-podman-ro-proxy:/run/forge-podman-ro-proxy:ro",
    ):
        assert mount in compose, f"missing directory mount: {mount}"

    assert "forge-podman-ro-proxy.sock:" not in compose, (
        "the compose example mounts the socket file again -- that is the "
        "mount that breaks on every proxy restart"
    )


def test_the_setup_script_checks_and_prints_the_same_path():
    """
    The setup script is what an operator copies the URL from. When it
    disagreed with the unit, nothing said so.
    """
    setup = SETUP.read_text()

    assert "${XDG_RUNTIME_DIR}/forge-podman-ro-proxy/sock" in setup
    assert "SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy/sock" in setup


def test_no_file_in_the_repository_still_names_the_old_socket():
    """
    A stale path in the docs is how someone ends up with a working
    proxy, a correct unit, and a container that cannot reach it.
    """
    # tests/ is excluded on purpose: this file quotes the old path in
    # its own docstring, and a regression test that cannot name the
    # thing it regressed against is worse than the duplication.
    stale = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "tests" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".yaml", ".yml", ".sh", ".service"}:
            continue
        if "forge-podman-ro-proxy.sock" in path.read_text(errors="ignore"):
            stale.append(str(path.relative_to(ROOT)))

    assert not stale, f"still naming the old socket path: {stale}"
