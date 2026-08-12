"""
Tests for the non-root container user (security audit, E-4).

Nothing here builds or runs an image -- that needs a real podman host
and is checked by hand against deploy/README.md's verification list.
What these hold is the pair of facts that make the change coherent,
because either one alone is worse than neither:

  - the image drops privileges (USER, non-root, after the steps that
    genuinely need root)
  - the documented runtime keeps the UID mapped to the host user
    (--userns=keep-id)

An image running as UID 1000 without keep-id maps to a subuid, which
loses the 0660 proxy sockets and the writable data directory. That
isn't a hardened container, it's a broken one, and it fails at
runtime on a machine none of the tests can reach -- so the coupling
is asserted here instead.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONTAINERFILE = (_ROOT / "Containerfile").read_text(encoding="utf-8")
_COMPOSE = (_ROOT / "deploy" / "compose.example.yaml").read_text(encoding="utf-8")
_DEPLOY_README = (_ROOT / "deploy" / "README.md").read_text(encoding="utf-8")


def _directive_lines(prefix: str) -> list[str]:
    return [
        line.strip()
        for line in _CONTAINERFILE.splitlines()
        if line.startswith(f"{prefix} ")
    ]


def test_the_image_declares_a_user():
    assert _directive_lines("USER"), (
        "Containerfile has no USER directive, so the serving process runs "
        "as container root and can rewrite /app/src/forge/ (audit E-4)."
    )


def test_the_declared_user_is_not_root():
    for line in _directive_lines("USER"):
        user = line.removeprefix("USER ").strip().strip('"')
        assert user.split(":")[0] not in ("0", "root"), line


def test_privileges_are_dropped_after_the_steps_that_need_them():
    """apt-get and pip must still run as root; only the CMD needs to be
    unprivileged. A USER placed too early breaks the build instead of
    the runtime, which is at least loud -- but it also tempts the next
    person to move it back rather than fix the ordering."""
    lines = _CONTAINERFILE.splitlines()
    user_index = next(i for i, line in enumerate(lines) if line.startswith("USER "))
    for i, line in enumerate(lines):
        if line.startswith("RUN ") and ("apt-get" in line or "pip install" in line):
            assert i < user_index, f"privileged step after USER: {line}"


def test_the_compose_example_keeps_the_uid_mapped_to_the_host_user():
    """The runtime half. Without this the image's UID 1000 lands on a
    subuid and loses both proxy sockets and ./data."""
    assert re.search(r"^\s*userns_mode:\s*\"keep-id", _COMPOSE, re.MULTILINE), (
        "deploy/compose.example.yaml must set userns_mode: keep-id -- a "
        "non-root image without it is a broken container, not a safer one."
    )


def test_the_escape_hatch_is_present_and_commented_out():
    """`user: "0:0"` restores the previous behaviour with no rebuild.
    It has to be shipped commented, and stay commented: uncommented it
    would silently undo the whole change."""
    assert '# user: "0:0"' in _COMPOSE
    assert not re.search(r"^\s*user:\s*\"0:0\"", _COMPOSE, re.MULTILINE)


def test_the_runtime_requirement_is_documented_next_to_the_run_commands():
    """A flag this load-bearing can't live only in a commit message --
    the person hitting the failure is reading deploy/README.md."""
    assert "--userns=keep-id" in _DEPLOY_README
    assert "Running non-root" in _DEPLOY_README
