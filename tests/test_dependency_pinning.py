"""
Forcing function for audit E-4: the requirement files stay pinned.

Pinning is only worth anything if it survives the next person (or the
next Evolution Runtime patch) adding a bare `some-lib` line. The
Containerfile installs from these files on every build, so an
unpinned line there is a silent, unreviewable change to the shipped
image. This test makes that a red suite instead.

Deliberately not checked here: whether the pins are current, or
whether they match what's installed in this venv. Freshness is a
judgement call for a human bumping versions on purpose; the shape of
the file is what a test can hold.
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REQUIREMENT_FILES = ["requirements.txt", "requirements-dev.txt"]


def _requirement_lines(filename: str) -> list[str]:
    text = (_ROOT / filename).read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("filename", _REQUIREMENT_FILES)
def test_every_requirement_is_pinned(filename):
    unpinned = [line for line in _requirement_lines(filename) if "==" not in line]
    assert not unpinned, (
        f"{filename} has unpinned requirements: {unpinned}. "
        "Pin them (see the header of the file for how to regenerate)."
    )


@pytest.mark.parametrize("filename", _REQUIREMENT_FILES)
def test_no_range_specifiers_alongside_the_pin(filename):
    """`foo==1.2,>=1.0` is a pin in appearance only -- it still lets
    the resolver move. Same for the `!=` / `~=` families."""
    loose = [
        line
        for line in _requirement_lines(filename)
        if any(op in line for op in (">=", "<=", "~=", "!=", ">", "<"))
    ]
    assert not loose, f"{filename} mixes range specifiers into pins: {loose}"


def test_the_files_are_not_empty():
    """Guards the guard: a truncated or renamed file would make both
    tests above pass on an empty list."""
    for filename in _REQUIREMENT_FILES:
        assert _requirement_lines(filename), f"{filename} has no requirements at all"
