"""
Tests that keep the documented Kernel section honest.

The capabilities example went stale once already: it was written when
Forge had 11 tools, and three landed on main before anyone looked at
it again. A doc example that drifts is worse than no example -- it
teaches the wrong output shape and nobody notices, because nothing
reads it.

So the example is pinned against the real command. Not character for
character (that would fail on every cosmetic tweak) but on the claims
a reader actually relies on: which capabilities exist, what each one
requires, and which ones the policy in the example blocks.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Searched across the docs rather than pinned to one file. The example
# used to live in README.md and now lives in docs/architecture.md, and
# a test that hardcoded either path would have failed on the move for
# a reason that has nothing to do with what it checks -- which is how
# a doc test gets deleted instead of fixed.
_DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]

_EXAMPLE = re.compile(
    r"```\n\$ forge capabilities\n(.*?)```",
    re.DOTALL,
)


def _docs_text() -> str:
    return "\n".join(p.read_text() for p in _DOCS if p.exists())


def _example_block() -> str:
    for path in _DOCS:
        if not path.exists():
            continue
        match = _EXAMPLE.search(path.read_text())
        if match:
            return match.group(1)
    raise AssertionError(
        "no page under README.md or docs/ shows a `forge capabilities` example any more"
    )


def _example_rows() -> dict[str, tuple[bool, str]]:
    """name -> (denied, requirements summary), parsed from the README."""
    rows = {}
    for line in _example_block().split("\n"):
        if not line.startswith((" x ", "   ")):
            continue
        parts = line.split()
        if not parts or parts[0] in ("CAPABILITY", "x"):
            denied = line.startswith(" x ")
            if not denied:
                continue
        denied = line.startswith(" x ")
        cells = line[3:].split("  ")
        cells = [c.strip() for c in cells if c.strip()]
        if len(cells) < 3:
            continue
        name, _provider, requires = cells[0], cells[1], "  ".join(cells[2:])
        rows[name] = (denied, requires)
    return rows


def test_the_documented_example_lists_every_shipped_capability():
    """
    The claim a reader checks first: is this the tool set I have?
    """
    import pkgutil

    import forge.tools as tools_pkg

    shipped = {
        m.name for m in pkgutil.iter_modules(tools_pkg.__path__) if m.name != "registry"
    }
    documented = set(_example_rows())

    assert documented == shipped, (
        "the documented `forge capabilities` example is out of date -- "
        f"missing: {sorted(shipped - documented)}, "
        f"stale: {sorted(documented - shipped)}"
    )


def test_the_documented_example_states_each_capability_s_real_requirements():
    """
    The second claim: does `research` really need the network? An
    example that misstates this is how someone concludes the policy
    flags do not work.
    """
    import importlib

    from forge.kernel.capability import Requirements

    wrong = {}
    for name, (_denied, documented) in _example_rows().items():
        mod = importlib.import_module(f"forge.tools.{name}")
        declared = getattr(mod, "REQUIREMENTS", None)
        if not isinstance(declared, Requirements):
            continue
        if declared.summary() != documented:
            wrong[name] = (documented, declared.summary())

    assert not wrong, f"documented requirements do not match the code: {wrong}"


def test_the_documented_example_blocks_exactly_what_its_stated_policy_blocks():
    """
    The example says `policy: denying network`. Every row it marks with
    an x must be one that policy really denies, and no other.
    """
    import importlib

    from forge.kernel.capability import Requirements

    block = _example_block()
    assert "policy: denying network" in block, (
        "this test pins the network-denied example; update it if the "
        "docs switch to a different policy"
    )

    for name, (denied, _requires) in _example_rows().items():
        mod = importlib.import_module(f"forge.tools.{name}")
        declared = getattr(mod, "REQUIREMENTS", None)
        if not isinstance(declared, Requirements):
            continue
        assert denied == declared.network, (
            f"{name}: the docs mark it {'denied' if denied else 'allowed'} "
            f"under a network-denying policy, but it declares "
            f"network={declared.network}"
        )


@pytest.mark.parametrize(
    "claim",
    [
        "kernel/",
        "capability.py",
        "registry.py",
        "policy.py",
    ],
)
def test_the_documented_file_tree_names_files_that_exist(claim):
    """
    A file tree that names a module nobody can open is the same failure
    as a stale example, and just as invisible.
    """
    assert claim in _docs_text()

    if claim.endswith(".py"):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "forge"
            / "kernel"
            / claim
        )
        assert path.exists(), f"the docs name {claim} but it does not exist"


def test_every_relative_link_in_the_docs_resolves():
    """
    The split from one 906-line README into docs/ moved every section
    one directory down, which silently turns `](ARCHITECTURE.md)` into
    a 404 and `](#some-anchor)` into a link to nothing. Neither breaks
    anything a test was watching, so nothing would have said so.

    Anchors are checked for existence of the target FILE only -- not
    the heading -- because heading slugs vary by renderer and pinning
    them here would fail for cosmetic edits.
    """
    import re as _re

    broken = []
    for path in _DOCS:
        if not path.exists():
            continue
        for target in _re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                continue  # same-page anchor
            file_part = target.split("#", 1)[0]
            if not file_part:
                continue
            if not (path.parent / file_part).exists():
                broken.append(f"{path.name} -> {target}")

    assert not broken, f"dead relative links in the docs: {broken}"
