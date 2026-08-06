"""
Git read-only tool.

Runs a curated set of git subcommands. All of them are genuinely
read-only — no commits, no push, no checkout, and nothing that
touches the working tree or the stash.

The allowed subcommands are hard-coded here (not configurable)
because git commands that mutate state require user confirmation, not
an LLM routing decision.

`stash` used to be on this list, under the same "read-only" heading.
It isn't: bare `git stash` is `git stash push`, which removes
uncommitted changes from the working tree, and `stash drop`/`clear`
destroy existing stashes. Nothing here was going to exfiltrate
anything through it — but on a project developed live, losing a
working tree costs a session. Removed.

A few flags are refused regardless of subcommand: git has options
that write files or execute programs (--output, --exec,
--upload-pack, --ext-diff...), which would turn a read-only
subcommand into a write or an execution.

To activate: ENABLED_TOOLS=chat,code,git in .env.local

Interface: run(content: str) -> str
  content is one of: status / diff / log / show / branch
  optionally with extra flags: "log --oneline -5"
"""

import shlex
import subprocess
from pathlib import Path

from forge.kernel.capability import Requirements
from forge.logger import log

# Read-only subcommands only, but each one is a subprocess.
REQUIREMENTS = Requirements(
    network=False,
    llm=False,
    mutates_workspace=False,
    spawns_process=True,
)


_ALLOWED_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "branch",
    "shortlog",
    "describe",
    "rev-parse",
}
# Refused wherever they appear in the arguments. Prefix-matched, so
# both "--output=x" and "--output x" are caught.
_BLOCKED_ARG_PREFIXES = (
    "--output",
    "--exec",
    "--upload-pack",
    "--receive-pack",
    "--ext-diff",
    "-o",
)
_TIMEOUT = 15
_MAX_OUTPUT_CHARS = 6_000

# Sane defaults appended when the user doesn't specify
_DEFAULTS = {
    "log": ["--oneline", "-15"],
    "diff": ["--stat"],
}

# Same minimal-env posture as tools/shell.py and tools/test.py, which
# this module was missing: without it the subprocess inherited the
# whole host environment, API_TOKEN and OPENROUTER_API_KEY included.
# GIT_* config vars are excluded along with everything else, which is
# what we want -- no ambient credential helper, no ambient identity.
_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": str(Path.home()),
    "TERM": "dumb",
    # Read-only commands should never prompt; if a repo somehow asks
    # for credentials, fail instead of hanging until the timeout.
    "GIT_TERMINAL_PROMPT": "0",
}


def _find_git_root() -> Path:
    """Walk up from cwd to find .git directory."""
    p = Path.cwd()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()  # fallback: use cwd even if not a git repo


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
            f"[error] git subcommand {subcommand!r} is not allowed.\nAllowed: {allowed}"
        )

    args = parts[1:] or _DEFAULTS.get(subcommand, [])

    for arg in args:
        if arg.startswith(_BLOCKED_ARG_PREFIXES):
            return (
                f"[error] argument {arg!r} is not allowed: it can write files "
                f"or execute a program, which would make a read-only "
                f"subcommand neither."
            )

    full_cmd = ["git", subcommand] + args
    cwd = _find_git_root()

    log.event("git.run", cmd=" ".join(full_cmd[:5]), cwd=str(cwd))

    try:
        result = subprocess.run(
            full_cmd,
            cwd=str(cwd),
            check=False,
            env=_ENV,
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
