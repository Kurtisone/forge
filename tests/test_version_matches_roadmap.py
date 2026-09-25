"""
The three places that name the current release agree.

`__version__`, the newest version row of docs/roadmap.md and the README's
"vX.Y is current" are each edited by hand, at different moments, by
different commits. v3.23 was tagged on a tree whose `__version__` still
said 3.22.0 and whose roadmap had no v3.23 row: nothing compared them,
so nothing could notice. A release name that means two things is what
docs/roadmap.md's own preamble says it split from ARCHITECTURE.md to
avoid.

Only major.minor is compared. A patch release (3.20.1) is a fix inside a
version that already has its row.
"""

import re
from pathlib import Path

import forge

_ROOT = Path(__file__).resolve().parent.parent

#: `| **v3.24** | done | ...` -- the version rows, not `Kernel L2` and
#: not `fix/dettes-v3.12`, which name something else.
_ROADMAP_ROW = re.compile(r"^\| \*\*v(\d+)\.(\d+)\*\* \|", re.MULTILINE)

_README_CURRENT = re.compile(r"\bv(\d+)\.(\d+) is current\b")


def _running() -> tuple[int, int]:
    major, minor, *_ = forge.__version__.split(".")
    return int(major), int(minor)


def test_the_newest_roadmap_row_is_the_running_version():
    text = (_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    rows = [(int(a), int(b)) for a, b in _ROADMAP_ROW.findall(text)]
    assert rows, "no `| **vX.Y** |` row found -- did the table's shape change?"
    assert max(rows) == _running(), (
        f"__version__ is {forge.__version__} but the newest roadmap row is "
        f"v{max(rows)[0]}.{max(rows)[1]}. Bump src/forge/__init__.py, or add "
        "the row -- one of the two is missing."
    )


def test_the_readme_names_the_running_version_as_current():
    text = (_ROOT / "README.md").read_text(encoding="utf-8")
    claims = [(int(a), int(b)) for a, b in _README_CURRENT.findall(text)]
    assert claims == [_running()], (
        f"README says {claims!r} is current; __version__ is {forge.__version__}"
    )
