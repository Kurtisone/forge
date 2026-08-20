"""
Allowlisted shell tool.

Executes commands in a subprocess. Two real protections, and one
convenience that is often mistaken for a third:

1. Allowlist: only commands in SHELL_ALLOWED_COMMANDS are accepted.
   This is the ONLY thing standing between router output and
   arbitrary execution, and it is worth exactly as much as the
   binaries on the list -- only parts[0] is checked, never the
   arguments, so a single interpreter on the list (python3, find,
   pip, xargs, ...) makes the whole allowlist decorative. See
   _SHELL_ALLOWLIST_DEFEATING in config.py; a warning is logged at
   import when one is configured.
2. Timeout: execution is hard-killed after SHELL_TIMEOUT seconds.
3. NOT a sandbox: commands run with cwd=WORKSPACE_DIR. This makes
   relative paths land in the workspace, which is convenient, but it
   confines nothing at all -- `cat /etc/passwd` and `grep -r . /`
   work exactly as they would anywhere else. An earlier version of
   this docstring claimed the cwd stopped commands escaping to the
   host filesystem; it does not, and reading it that way is how an
   allowlist gets widened "safely".

The environment passed to the subprocess is minimal (PATH, HOME,
PYTHONPATH) -- no credentials, no tokens, no host env variables.
Note HOME still points at the real home directory, so a command able
to read files can read ~/.ssh: another reason the allowlist matters
more than the cwd.

To activate: ENABLED_TOOLS=chat,code,shell in .env.local
To customise: SHELL_ALLOWED_COMMANDS=ls,cat,grep

Interface: run(content: str) -> str
  content is a plain command string: "ls -la" / "grep -n foo bar.py"
"""

import shlex
import subprocess
from pathlib import Path

from forge.config import (
    _SHELL_ALLOWLIST_DEFEATING,
    SHELL_ALLOWED_COMMANDS,
    SHELL_TIMEOUT,
    WORKSPACE_DIR,
)
from forge.kernel.capability import Requirements
from forge.logger import log

# Intentionally the conservative default, spelled out: the allowlist
# is SHELL_ALLOWED_COMMANDS, so what a command may reach is a
# deployment-time choice and cannot be known statically here.
REQUIREMENTS = Requirements()


_MAX_OUTPUT_CHARS = 8_000

# Logged once, at import, rather than on every run: this is a
# configuration fact, not a per-call event, and burying it in the
# output of each command is how people learn to ignore it.
_defeating = sorted(SHELL_ALLOWED_COMMANDS & _SHELL_ALLOWLIST_DEFEATING)
if _defeating:
    log.warning(
        "shell: SHELL_ALLOWED_COMMANDS contains %s -- each of these can execute "
        "arbitrary commands via its own arguments, so the allowlist no longer "
        "restricts anything. Intentional on a trusted local box; never on an "
        "instance reachable from the network.",
        ", ".join(_defeating),
    )


def _safe_cwd() -> Path:
    cwd = Path(WORKSPACE_DIR).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def run(content: str) -> str:
    command = content.strip()
    if not command:
        return "[error] empty command"

    # Parse and check the base command name
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"[error] could not parse command: {e}"

    base = parts[0]
    if base not in SHELL_ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(SHELL_ALLOWED_COMMANDS))
        return (
            f"[error] command {base!r} is not in the allowlist.\n"
            f"Allowed: {allowed}\n"
            f"Add it to SHELL_ALLOWED_COMMANDS in .env.local to enable it."
        )

    cwd = _safe_cwd()
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(Path(WORKSPACE_DIR).resolve() / "src"),
        "TERM": "dumb",
    }

    log.event("shell.run", command=command[:80], cwd=str(cwd))

    try:
        result = subprocess.run(
            parts,
            cwd=str(cwd),
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "shell: command timed out after %ds: %s", SHELL_TIMEOUT, command[:60]
        )
        return f"[error] command timed out after {SHELL_TIMEOUT}s"
    except FileNotFoundError:
        return f"[error] executable not found: {base!r}"
    except OSError as e:
        return f"[error] OS error: {e}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    output_parts = []
    if stdout:
        output_parts.append(stdout)
    if stderr:
        output_parts.append(f"[stderr]\n{stderr}")
    if result.returncode != 0 and not stderr:
        output_parts.append(f"[exit code {result.returncode}]")

    output = "\n".join(output_parts) if output_parts else "[no output]"

    if len(output) > _MAX_OUTPUT_CHARS:
        output = (
            output[:_MAX_OUTPUT_CHARS]
            + f"\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"
        )

    log.event("shell.done", returncode=result.returncode, output_chars=len(output))
    return output
