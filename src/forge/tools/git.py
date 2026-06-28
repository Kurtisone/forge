"""
Git read-only tool.

Runs a curated set of git commands in the current working directory.
All operations are read-only — no commits, no push, no checkout.

The allowed git subcommands are hard-coded here (not configurable)
because git commands that mutate state require user confirmation, not
an LLM routing decision.

To activate: ENABLED_TOOLS=chat,code,git in .env.local

Interface: run(content: str) -> str
  content is one of: status / diff / log / show / branch / stash
  optionally with extra flags: "log --oneline -5"
"""

import shlex
import subprocess
from pathlib import Path

from forge.logger import log

_ALLOWED_SUBCOMMANDS = {
    "status", "diff", "log", "show", "branch",
    "stash", "shortlog", "describe", "rev-parse",
}
_TIMEOUT = 15
_MAX_OUTPUT_CHARS = 6_000

# Sane defaults appended when the user doesn't specify
_DEFAULTS = {
    "log":  ["--oneline", "-15"],
    "diff": ["--stat"],
}


def _find_git_root() -> Path:
    """Walk up from cwd to find .git directory."""
    p = Path.cwd()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()   # fallback: use cwd even if not a git repo


def run(content: str) -> str:
    command = content.strip()
    if not command:
        return "[error] empty git command"

    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"[error] could not parse command: {e}"

    subcommand = parts[0]
    if subcommand not in _ALLOWED_SUBCOMMANDS:
        allowed = ", ".join(sorted(_ALLOWED_SUBCOMMANDS))
        return (
            f"[error] git subcommand {subcommand!r} is not allowed.\n"
            f"Allowed: {allowed}"
        )

    args = parts[1:] or _DEFAULTS.get(subcommand, [])
    full_cmd = ["git", subcommand] + args
    cwd = _find_git_root()

    log.event("git.run", cmd=" ".join(full_cmd[:5]), cwd=str(cwd))

    try:
        result = subprocess.run(
            full_cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"[error] git command timed out after {_TIMEOUT}s"
    except FileNotFoundError:
        return "[error] git is not installed or not in PATH"
    except OSError as e:
        return f"[error] OS error: {e}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        return f"[git error]\n{stderr or stdout}"

    output = stdout or "[no output]"
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"

    log.event("git.done", lines=output.count("\n"))
    return output
