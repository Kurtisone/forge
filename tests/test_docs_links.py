"""
Every relative link in the documentation, and whether it resolves.

WHY THIS IS A TEST AND NOT A HARNESS. bench/ answers questions that
have numbers and need a store or a server. This one has a single
correct answer, needs neither, and either passes or fails -- which is
the definition of something CI should run. As a test rather than a
workflow step it also runs locally, before the commit, in the same
command everything else runs in.

WHAT IT CATCHES THAT A READER DOES NOT. A broken link is invisible
until somebody clicks it, and a broken ANCHOR is worse: GitHub
silently lands the reader at the top of the right page, which reads as
"the section was moved or removed" rather than as a typo. Both rot
from ordinary edits -- renaming a heading is enough.

It was written the day docs/memory.md was split in two and 543 lines
changed file, which is exactly the edit that breaks anchors in bulk.
The split itself was fine; the next one will not necessarily be.

THE ANCHOR RULE IS GITHUB'S, APPROXIMATELY. Lowercase, drop everything
that is not a word character, a space or a hyphen, then spaces to
hyphens. Close enough for the links this repository writes, and a
heading it gets wrong fails LOUDLY here rather than quietly in a
browser. What it deliberately does not model is GitHub's `-1` suffix
for duplicate headings: two identical headings on one page are worth
failing over anyway.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Markdown files that are part of the documentation, wherever they
#: live. deploy/README.md is included because usage.md links into it,
#: which is the kind of cross-directory link that rots unnoticed.
DOCS = [
    ROOT / "README.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "SECURITY.md",
    ROOT / "deploy" / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_NOT_IN_ANCHOR = re.compile(r"[^\w\- ]")


def headings(path: pathlib.Path) -> list[str]:
    """
    The heading lines of a page, code fences excluded.

    The exclusion is not a refinement. Every `bash` block in usage.md
    opens with a shell comment -- `# Review a file` -- and the first
    version of this file read four of them as headings, which both
    invents anchors that do not exist and reports duplicates that are
    not there. A test that is wrong about what a heading is fails on
    correct documentation, and a test that fails on correct
    documentation gets deleted.
    """
    out, fenced = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("#"):
            out.append(line)
    return out


def anchor_for(heading: str) -> str:
    return _NOT_IN_ANCHOR.sub("", heading.lstrip("#").strip()).lower().replace(" ", "-")


def _anchors(path: pathlib.Path) -> set[str]:
    """The anchors GitHub would generate for one page's headings."""
    return {anchor_for(h) for h in headings(path)}


def _links(path: pathlib.Path):
    """(target, file part, anchor) for every link that points inside the repo."""
    for target in _LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, anchor = target.partition("#")
        yield target, file_part, anchor


def test_every_documented_file_exists():
    missing = []
    for doc in DOCS:
        for target, file_part, _ in _links(doc):
            if not file_part:
                continue
            if not (doc.parent / file_part).resolve().exists():
                missing.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not missing, "links to files that do not exist:\n" + "\n".join(missing)


def test_every_anchor_exists_on_the_page_it_points_at():
    """
    The half a reader cannot see failing: GitHub lands them at the top
    of the page instead of at the section, which reads as a section
    that was removed.
    """
    broken = []
    for doc in DOCS:
        for target, file_part, anchor in _links(doc):
            if not anchor:
                continue
            page = (doc.parent / file_part).resolve() if file_part else doc
            if not page.exists() or page.suffix != ".md":
                continue
            if anchor not in _anchors(page):
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, "links to anchors that do not exist:\n" + "\n".join(broken)


def test_no_page_has_two_headings_with_the_same_anchor():
    """
    GitHub disambiguates these with a `-1` suffix, so a link to either
    one silently reaches the first. Worth failing over rather than
    modelling: the fix is to rename a heading, and the alternative is a
    link whose target depends on heading order.
    """
    clashes = []
    for doc in DOCS:
        seen = set()
        for heading in headings(doc):
            anchor = anchor_for(heading)
            if anchor in seen:
                clashes.append(f"{doc.relative_to(ROOT)}: {heading.strip()}")
            seen.add(anchor)
    assert not clashes, "two headings sharing one anchor:\n" + "\n".join(clashes)


def test_the_documentation_index_lists_every_page():
    """
    A page nobody links to is a page nobody reads. docs/README.md is
    the one table a reader is guaranteed to see, and a new page that
    misses it is discoverable only by `ls`.
    """
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    unlisted = [
        page.name
        for page in sorted((ROOT / "docs").glob("*.md"))
        if page.name != "README.md" and f"]({page.name})" not in index
    ]
    assert not unlisted, f"pages missing from docs/README.md: {unlisted}"
