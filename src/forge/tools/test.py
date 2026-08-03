"""
Sandboxed test/lint tool.

Runs test and lint commands confined to WORKSPACE_DIR, via a
dedicated allowlist independent from the general-purpose shell tool
(TEST_ALLOWED_COMMANDS, default: pytest,ruff). Deliberately separate
from tools/shell.py: "run the tests" / "lint this" should be
first-class router intents with their own narrow, purpose-built
allowlist, rather than the router constructing a raw shell command
that happens to also be allowed by SHELL_ALLOWED_COMMANDS.

Same three protection layers as shell.py:
1. Allowlist: only runners in TEST_ALLOWED_COMMANDS are accepted.
2. Timeout: execution is hard-killed after TEST_TIMEOUT seconds.
3. Working directory: runs with cwd=WORKSPACE_DIR so relative paths
   cannot escape to the host filesystem.

The allowed runner's executable is resolved via shutil.which() against
the real process PATH, not the minimal one the subprocess itself runs
with -- a pip-installed console script like pytest/ruff can land
anywhere depending on how Python was set up (a venv, --user, a CI
runner's own hostedtoolcache, …), unlike coreutils which reliably
live in /usr/bin. Confirmed live: this tool passed in one environment
and failed with "executable not found" in GitHub Actions CI before
this was fixed.

To activate: ENABLED_TOOLS=chat,code,test in .env.local
To customise: TEST_ALLOWED_COMMANDS=pytest,ruff

Interface: run(content: str) -> str
  content is "<runner> <args>", e.g.:
    "pytest tests/test_graph.py"
    "pytest tests/ -k test_shell"
    "ruff check src/forge/graph.py"
"""

import shlex
import shutil
import subprocess
from pathlib import Path

from forge.config import TEST_ALLOWED_COMMANDS, TEST_TIMEOUT, WORKSPACE_DIR
from forge.logger import log

_MAX_OUTPUT_CHARS = 8_000


def _safe_cwd() -> Path:
    cwd = Path(WORKSPACE_DIR).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def run(content: str) -> str:
    command = content.strip()
    if not command:
        return "[error] empty test command"

    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"[error] could not parse command: {e}"

    runner = parts[0]
    if runner not in TEST_ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(TEST_ALLOWED_COMMANDS))
        return (
            f"[error] test runner {runner!r} is not in the allowlist.\n"
            f"Allowed: {allowed}\n"
            f"Add it to TEST_ALLOWED_COMMANDS in .env.local to enable it."
        )

    resolved = shutil.which(runner)
    if resolved is None:
        return (
            f"[error] executable not found: {runner!r} (not on PATH)\n"
            f"Is {runner!r} installed in this environment?"
        )

    cwd = _safe_cwd()
    # Minimal env for the subprocess itself (no leaked host secrets/
    # tokens) -- but the LOOKUP of where the runner actually lives
    # uses the real process PATH via shutil.which() above, not this
    # restricted one. A hardcoded PATH guess (e.g. "/usr/bin:/bin")
    # works for coreutils but not for pip-installed console scripts
    # like pytest/ruff, which can land anywhere depending on how
    # Python was set up (a venv, --user, a CI runner's hostedtoolcache,
    # …) -- confirmed live: this exact tool passed in one environment
    # and failed with "executable not found" in GitHub Actions CI
    # until resolved this way.
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": str(Path(WORKSPACE_DIR).resolve() / "src"),
        "TERM": "dumb",
    }

    log.event("test.run", command=command[:80], cwd=str(cwd))

    try:
        result = subprocess.run(
            [resolved, *parts[1:]],
            cwd=str(cwd),
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.warning("test: command timed out after %ds: %s", TEST_TIMEOUT, command[:60])
        return f"[error] command timed out after {TEST_TIMEOUT}s"
    except FileNotFoundError:
        return f"[error] executable not found: {runner!r}"
    except OSError as e:
        return f"[error] OS error: {e}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    output_parts = []
    if stdout:
        output_parts.append(stdout)
    if stderr:
        output_parts.append(f"[stderr]\n{stderr}")
    if result.returncode != 0 and not output_parts:
        output_parts.append(f"[exit code {result.returncode}]")

    output = "\n".join(output_parts) if output_parts else "[no output]"

    if len(output) > _MAX_OUTPUT_CHARS:
        output = (
            output[:_MAX_OUTPUT_CHARS]
            + f"\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"
        )

    log.event("test.done", returncode=result.returncode, output_chars=len(output))
    return output
