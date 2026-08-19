"""
Tests that keep the README's Kernel section honest.

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

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"

_EXAMPLE = re.compile(
    r"```\n\$ forge capabilities\n(.*?)```",
    re.DOTALL,
)


def _example_block() -> str:
    match = _EXAMPLE.search(README.read_text())
    assert match, "the README no longer shows a `forge capabilities` example"
    return match.group(1)


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


def test_the_readme_example_lists_every_shipped_capability():
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
        "the README's `forge capabilities` example is out of date -- "
        f"missing: {sorted(shipped - documented)}, "
        f"stale: {sorted(documented - shipped)}"
    )


def test_the_readme_example_states_each_capability_s_real_requirements():
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

    assert not wrong, f"README requirements do not match the code: {wrong}"


def test_the_readme_example_blocks_exactly_what_its_stated_policy_blocks():
    """
    The example says `policy: denying network`. Every row it marks with
    an x must be one that policy really denies, and no other.
    """
    import importlib

    from forge.kernel.capability import Requirements

    block = _example_block()
    assert "policy: denying network" in block, (
        "this test pins the network-denied example; update it if the "
        "README switches to a different policy"
    )

    for name, (denied, _requires) in _example_rows().items():
        mod = importlib.import_module(f"forge.tools.{name}")
        declared = getattr(mod, "REQUIREMENTS", None)
        if not isinstance(declared, Requirements):
            continue
        assert denied == declared.network, (
            f"{name}: README marks it {'denied' if denied else 'allowed'} "
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
def test_the_readme_file_tree_names_files_that_exist(claim):
    """
    A file tree that names a module nobody can open is the same failure
    as a stale example, and just as invisible.
    """
    assert claim in README.read_text()

    if claim.endswith(".py"):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "forge"
            / "kernel"
            / claim
        )
        assert path.exists(), f"README names {claim} but it does not exist"
