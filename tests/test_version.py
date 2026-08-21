"""
Tests that keep the version honest.

Before this, `version` sat in pyproject.toml and again as a literal in
api.py's FastAPI() call. Both read "3.3.0" while the working version
was 3.13. Nothing failed, because nothing compared them -- the number
is only ever read by OpenAPI and by whoever builds a wheel, and
neither of those has an opinion.

The fix is structural (pyproject reads the attribute, api.py imports
it), so these tests mostly guard the structure: they fail if someone
reintroduces a second copy, not if the number changes.
"""

import pathlib
import re
import tomllib

import forge

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_pyproject_takes_its_version_from_the_package():
    """
    The claim: there is no second copy to drift.

    A literal `version = "..."` under [project] would silently win over
    the attribute and put us back where we started.
    """
    data = _pyproject()

    assert "version" not in data["project"], (
        "pyproject.toml declares a literal version again -- that is the "
        "second copy this test exists to prevent; use "
        'dynamic = ["version"] and point it at forge.__version__'
    )
    assert "version" in data["project"].get("dynamic", []), (
        "pyproject.toml must declare the version dynamic"
    )
    assert (
        data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "forge.__version__"
    )


def test_the_version_attribute_is_a_plain_literal():
    """
    setuptools parses __init__.py statically -- it does not import it.
    A computed version (importlib.metadata, a git call, an f-string)
    parses fine here and breaks the build backend instead, which is a
    far worse place to find out.
    """
    source = (ROOT / "src" / "forge" / "__init__.py").read_text()
    match = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)

    assert match, (
        "forge.__version__ must be a plain string literal on its own "
        "line -- setuptools reads this file without executing it"
    )
    assert match.group(1) == forge.__version__


def test_the_api_serves_that_same_version():
    """
    /openapi.json is the only place this number is published, so it is
    the copy that used to be wrong in public.
    """
    from forge.api import app

    assert app.version == forge.__version__


def test_the_version_looks_like_a_release_number():
    """
    Not a style rule -- git tags are cut from this string, and the
    existing tag history already contains one entry (v2.1) that skipped
    the patch component and sorts differently from its neighbours.
    """
    assert re.fullmatch(r"\d+\.\d+\.\d+", forge.__version__), (
        f"expected MAJOR.MINOR.PATCH, got {forge.__version__!r}"
    )
