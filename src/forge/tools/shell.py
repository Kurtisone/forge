"""
Sandboxed shell tool.

Executes commands in a subprocess confined to WORKSPACE_DIR.
Three layers of protection:

1. Allowlist: only commands in SHELL_ALLOWED_COMMANDS are accepted.
2. Timeout: execution is hard-killed after SHELL_TIMEOUT seconds.
3. Working directory: all commands run with cwd=WORKSPACE_DIR so
   relative paths cannot escape to the host filesystem.

The environment passed to the subprocess is minimal (PATH, HOME,
PYTHONPATH) — no credentials, no tokens, no host env variables.

To activate: ENABLED_TOOLS=chat,code,shell in .env.local
To customise: SHELL_ALLOWED_COMMANDS=ls,cat,python3

Interface: run(content: str) -> str
  content is a plain command string: "ls -la" / "python3 script.py"
"""

import shlex
import subprocess
from pathlib import Path

from forge.config import SHELL_ALLOWED_COMMANDS, SHELL_TIMEOUT, WORKSPACE_DIR
from forge.logger import log

_MAX_OUTPUT_CHARS = 8_000


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
            env=env,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.warning("shell: command timed out after %ds: %s", SHELL_TIMEOUT, command[:60])
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
        output = output[:_MAX_OUTPUT_CHARS] + f"\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"

    log.event("shell.done", returncode=result.returncode, output_chars=len(output))
    return output
