"""
Sandboxed file tool.

Provides read / write / list operations strictly confined to
WORKSPACE_DIR (default: data/workspace). Path traversal is rejected
at the boundary: any path that resolves outside the workspace raises
a PermissionError before any filesystem operation is attempted.

Interface (consistent with all Forge tools):
    run(content: str) -> str

content is a JSON instruction:
    {"action": "read",  "path": "relative/path.py"}
    {"action": "write", "path": "relative/out.txt", "content": "..."}
    {"action": "list",  "path": "optional/subdir"}   # default: root

Writing to a path that already exists returns a real diff (computed
with difflib, not asked of the model) instead of a bare confirmation
-- "improve this file" then shows what actually changed, not the
whole file again. Writing a brand-new path still returns a plain
"[ok] written N bytes" confirmation, since there's nothing to diff
against.

To activate this tool add it to ENABLED_TOOLS in .env.local:
    ENABLED_TOOLS=chat,code,files
"""

import difflib
import json
from pathlib import Path

from forge.config import WORKSPACE_DIR
from forge.logger import log

_MAX_READ_BYTES = 64 * 1024  # 64 KB — refuse to read huge files


def _safe_path(relative: str) -> Path:
    """
    Resolve *relative* against WORKSPACE_DIR and verify it stays inside.
    Raises PermissionError on any traversal attempt.
    """
    workspace = Path(WORKSPACE_DIR).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    target = (workspace / relative).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        raise PermissionError(f"path {relative!r} escapes workspace {str(workspace)!r}")
    return target


def _action_read(path_str: str) -> str:
    target = _safe_path(path_str)
    if not target.exists():
        return f"[error] file not found: {path_str}"
    if not target.is_file():
        return f"[error] not a file: {path_str}"
    size = target.stat().st_size
    if size > _MAX_READ_BYTES:
        return f"[error] file too large ({size} bytes > {_MAX_READ_BYTES} limit)"
    content = target.read_text(encoding="utf-8", errors="replace")
    log.event("files.read", path=path_str, bytes=size)
    return content


def _action_write(path_str: str, content: str) -> str:
    target = _safe_path(path_str)
    existed = target.exists()
    # Read the old content up front, before it's overwritten -- if
    # this fails or the old file is too large to diff safely, fall
    # back to the plain confirmation rather than losing the write.
    old_content = None
    if existed:
        try:
            if target.stat().st_size <= _MAX_READ_BYTES:
                old_content = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            old_content = None

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    log.event("files.write", path=path_str, bytes=len(content))

    if old_content is None:
        return f"[ok] written {len(content)} bytes to {path_str}"

    # A real diff, computed here rather than asked of the model: a
    # small local model asked to "improve this file" tends to just
    # regenerate the whole thing as its answer even when only a few
    # lines actually changed, and asking it to hand-format diff syntax
    # itself is failure-prone. difflib is deterministic and always
    # correct, independent of what the model actually produced.
    diff_lines = list(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"{path_str} (avant)",
            tofile=f"{path_str} (après)",
        )
    )
    if not diff_lines:
        return f"[ok] {path_str} inchangé ({len(content)} octets, contenu identique)"

    diff_text = "".join(diff_lines)
    if not diff_text.endswith("\n"):
        diff_text += "\n"
    return (
        f"[ok] {path_str} mis à jour ({len(content)} octets)\n\n```diff\n{diff_text}```"
    )


def _action_list(path_str: str = ".") -> str:
    target = _safe_path(path_str)
    if not target.exists():
        return f"[error] directory not found: {path_str}"
    if not target.is_dir():
        return f"[error] not a directory: {path_str}"
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = []
    for entry in entries:
        prefix = "  " if entry.is_file() else "/ "
        lines.append(f"{prefix}{entry.name}")
    log.event("files.list", path=path_str, entries=len(lines))
    return "\n".join(lines) if lines else "(empty directory)"


def run(content: str) -> str:
    """
    Execute a file operation described by a JSON instruction.

    Expected shapes:
        {"action": "read",  "path": "src/forge/main.py"}
        {"action": "write", "path": "notes.txt", "content": "..."}
        {"action": "list",  "path": "."}
    """
    try:
        instruction = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return f"[error] files tool expects JSON, got: {content[:80]!r}"

    action = instruction.get("action", "").strip().lower()
    path = instruction.get("path", "").strip()

    if not action:
        return "[error] missing 'action' field (read / write / list)"

    try:
        if action == "read":
            if not path:
                return "[error] 'read' requires a 'path' field"
            return _action_read(path)

        if action == "write":
            if not path:
                return "[error] 'write' requires a 'path' field"
            file_content = instruction.get("content", "")
            return _action_write(path, file_content)

        if action == "list":
            return _action_list(path or ".")

        return f"[error] unknown action {action!r} (use: read / write / list)"

    except PermissionError as e:
        log.error("files tool: permission denied: %s", e)
        return f"[error] permission denied: {e}"
    except OSError as e:
        log.error("files tool: OS error: %s", e)
        return f"[error] filesystem error: {e}"
