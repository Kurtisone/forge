"""
Dispatchable wrapper around forge.graphs.review.

Forge's UI is a single conversational page, zero tabs -- there is no
dedicated review form and there won't be one. This module is what
lets a plain chat message ("relis ce fichier", "vérifie ce fichier et
ses tests") trigger graphs.review.run() through the normal
router → dispatch path, the same way `files` and `memory` already do
for their own multi-field operations.

Both file_path and test_path are confined to WORKSPACE_DIR here --
same threat model and same confinement as tools/files.py, since this
input comes from router output on untrusted chat text. This is
deliberately NOT done inside graphs/review.py itself: that function
is also called directly by the CLI (a human typing a real path,
trusted the same way shell access is) and by the /review API
endpoint (which reviews its own tempfile, outside the workspace by
design). Confinement belongs at this dispatch boundary, not in the
shared graph.

A leading "/" is stripped rather than rejected: "/hello.go" and
"hello.go" both mean the same file at the workspace root. This is
also what actually closes the escape, not just what's convenient --
Path(workspace) / "/etc/passwd" would otherwise silently discard the
workspace prefix entirely (a pathlib join with an absolute right-hand
side replaces the left side), so stripping first is what guarantees
the join always stays relative.

Interface: run(content: str) -> str
  content is a JSON string:
    {"file_path": "...", "question": "...", "test_path": "..."}
  question and test_path are optional.

To activate: ENABLED_TOOLS=chat,code,review in .env.local
"""

import json
from pathlib import Path

from forge.config import WORKSPACE_DIR
from forge.graphs.review import run as review_run
from forge.kernel.capability import Requirements
from forge.tool_payload import loads_payload

# A filename component cannot exceed 255 bytes on ext4/btrfs, so a
# longer one is not a path that happens not to exist -- it is not a
# path. Checked before the filesystem is touched because the OS error
# for it is not a quiet False: see _looks_like_a_path.
_MAX_PATH_CHARS = 255


def _looks_like_a_path(relative: str) -> str | None:
    """
    Reject strings that cannot be a path at all. Returns a reason, or
    None when the string is plausible.

    Seen live on 2026-08-17: asked to review a pasted workout routine,
    the router chose review and put the ENTIRE pasted text into
    file_path -- 640 characters with newlines. Path.exists() does not
    return False for that. ENAMETOOLONG is not in pathlib's ignored
    errno list, so exists() RAISES, the node blew up before its own
    error handling, and the run surfaced as "returned empty output".

    The lot-3 path grounding guard cannot help here, and that is worth
    understanding rather than patching around: it checks PROVENANCE --
    does this path appear in the conversation -- and this one did,
    because it was the conversation. Grounding says where a string came
    from, not whether it is a path. The two checks are complementary,
    not redundant.
    """
    if not relative.strip():
        return "empty path"
    if "\n" in relative or "\r" in relative:
        return "path contains a line break"
    if len(relative) > _MAX_PATH_CHARS:
        return f"path is {len(relative)} chars, over the {_MAX_PATH_CHARS} limit"
    return None


# Reads a file and calls the LLM. Running the optional tests is
# delegated to the `test` tool, which declares its own profile.
REQUIREMENTS = Requirements(
    network=False,
    llm=True,
    mutates_workspace=False,
    spawns_process=False,
)


def _safe_workspace_path(relative: str) -> Path:
    """Resolve *relative* against WORKSPACE_DIR and verify it stays
    inside. Raises PermissionError on any traversal attempt that
    survives the leading-slash strip (e.g. "../../etc/passwd"), and
    ValueError on a string that is not a path at all."""
    reason = _looks_like_a_path(relative)
    if reason is not None:
        raise ValueError(reason)

    workspace = Path(WORKSPACE_DIR).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    target = (workspace / relative.lstrip("/")).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        raise PermissionError(
            f"path {relative!r} escapes workspace {str(workspace)!r}"
        ) from None
    return target


def run(content: str) -> str:
    try:
        payload = loads_payload(content, "review")
    except (json.JSONDecodeError, TypeError):
        return '[error] review content must be JSON: {"file_path": "..."}'

    if not isinstance(payload, dict):
        return '[error] review content must be a JSON object: {"file_path": "..."}'

    file_path = payload.get("file_path")
    if not file_path:
        return "[error] missing 'file_path' in review request"

    try:
        safe_file_path = _safe_workspace_path(file_path)
    except PermissionError as e:
        return f"[error] {e}"
    except ValueError as e:
        # Says what was wrong AND what the field is for. The router put
        # a document here once; a bare "invalid path" would leave it to
        # guess, and it guesses by repeating itself.
        return (
            f"[error] {e}. 'file_path' must be the NAME of a file in the "
            "workspace, not its contents."
        )

    question = payload.get("question") or "Que peut-on améliorer ?"
    test_path = payload.get("test_path")
    safe_test_path = None
    if test_path:
        try:
            safe_test_path = str(
                _safe_workspace_path(test_path).relative_to(
                    Path(WORKSPACE_DIR).resolve()
                )
            )
        except (PermissionError, ValueError) as e:
            return f"[error] {e}"

    return review_run(str(safe_file_path), question=question, test_path=safe_test_path)
