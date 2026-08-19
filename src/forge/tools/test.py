"""
Test/lint tool. NOT a sandbox -- read the second section.

Runs test and lint commands in WORKSPACE_DIR, via a dedicated
allowlist independent from the general-purpose shell tool
(TEST_ALLOWED_COMMANDS, default: pytest,ruff). Deliberately separate
from tools/shell.py: "run the tests" / "lint this" should be
first-class router intents with their own narrow, purpose-built
allowlist, rather than the router constructing a raw shell command
that happens to also be allowed by SHELL_ALLOWED_COMMANDS.

What this tool actually guarantees:
1. Allowlist: only runners in TEST_ALLOWED_COMMANDS are accepted.
2. Timeout: execution is hard-killed after TEST_TIMEOUT seconds.
3. Path arguments stay inside WORKSPACE_DIR: an absolute path or a
   `..` climbing out is rejected before anything runs, so this tool
   can't be used to lint /etc or collect tests from the host.
4. Minimal subprocess env (PATH, HOME, PYTHONPATH) -- no host
   credentials or tokens are passed down.

What it does NOT guarantee (audit E-1):

    Running pytest means executing the Python code in the workspace.
    That is the whole point of a test runner, and there is no version
    of it that isn't arbitrary code execution. `pytest test_x.py`
    imports test_x.py and runs whatever is at module level; pytest
    also auto-loads conftest.py from the rootdir before collecting
    anything, so even an invocation that names only a specific,
    innocuous file executes a conftest.py sitting next to it.

    So if `files` and `test` are both enabled, together they are
    equivalent to `shell`, whatever SHELL_ALLOWED_COMMANDS says:
    write a file, then run it. The allowlist here restricts which
    binary starts, not what that binary then executes. Point 3 above
    bounds WHERE that code can come from -- the workspace -- it does
    not stop it running.

    Making that safe needs isolation this process doesn't have (a
    disposable container per run). Until then the honest statement is
    the one above, and a warning is logged at import when both tools
    are enabled. See SECURITY.md.

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

from forge.config import (
    ENABLED_TOOLS,
    TEST_ALLOWED_COMMANDS,
    TEST_TIMEOUT,
    WORKSPACE_DIR,
)
from forge.logger import log

_MAX_OUTPUT_CHARS = 8_000

# Logged once at import, same reasoning as shell.py's allowlist
# tripwire: this is a configuration fact, and repeating it into every
# run's output is how people learn to scroll past it.
if {"files", "test"} <= ENABLED_TOOLS:
    log.warning(
        "test: 'files' and 'test' are both enabled -- together they are "
        "equivalent to 'shell' regardless of SHELL_ALLOWED_COMMANDS "
        "(write a file, then run it; pytest executes workspace code by "
        "design, and auto-loads conftest.py before collection). "
        "Reasonable on a trusted local box, never on an instance "
        "reachable from the network. See SECURITY.md."
    )


def _safe_cwd() -> Path:
    cwd = Path(WORKSPACE_DIR).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _escaping_arg(arg: str, workspace: Path) -> bool:
    """
    Would this argument point outside the workspace?

    Flags are skipped: "--tb=short" is not a path, and a value that
    follows a flag ("-k", "not slow") resolves harmlessly inside the
    workspace anyway. What this catches is the shape that matters --
    an absolute path, or a relative one climbing out with "..".

    Note the asymmetry with files.py's _safe_path, which reinterprets
    a leading "/" as the workspace root rather than rejecting it.
    That call was made because the router genuinely emits "/hello.go"
    meaning the workspace file. Here an absolute path is far more
    likely to be exactly what it looks like -- "ruff check /etc" --
    and rewriting it silently would turn a refusal into a surprise.
    """
    if arg.startswith("-"):
        return False
    if Path(arg).is_absolute():
        return True
    try:
        (workspace / arg).resolve().relative_to(workspace)
    except ValueError:
        return True
    return False


def _looks_like_a_path(token: str) -> bool:
    """Is this first token a file to act ON, rather than the runner?"""
    return "/" in token or bool(Path(token).suffix)


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
        # A path in first position is a different mistake from a
        # forbidden binary, and telling someone to add
        # "tests/test_x.py" to TEST_ALLOWED_COMMANDS is advice that
        # cannot work. Observed live 2026-08-19: "Lance les tests dans
        # tests/test_inexistant_xyz.py" routed to
        # {"tool":"test","content":"tests/test_inexistant_xyz.py"} and
        # got exactly that answer.
        #
        # Named, never repaired. Prepending a default runner would
        # mean guessing between pytest and ruff -- the runner IS the
        # difference between the two intents this tool serves, and it
        # is not recoverable from the payload. The router now has a
        # description and two examples for this tool (it had neither);
        # this message is what happens when they lose.
        if _looks_like_a_path(runner):
            return (
                f"[error] {runner!r} is a path, not a test runner.\n"
                f"content must start with the runner: e.g. "
                f'"pytest {runner}" to test it, "ruff check {runner}" '
                "to lint it.\n"
                f"Allowed runners: {allowed}"
            )
        return (
            f"[error] test runner {runner!r} is not in the allowlist.\n"
            f"Allowed: {allowed}\n"
            f"Add it to TEST_ALLOWED_COMMANDS in .env.local to enable it."
        )

    workspace = _safe_cwd()
    for arg in parts[1:]:
        if _escaping_arg(arg, workspace):
            return (
                f"[error] argument {arg!r} points outside the workspace.\n"
                f"This tool only runs against {str(workspace)!r}; use a "
                "path relative to it."
            )

    resolved = shutil.which(runner)
    if resolved is None:
        return (
            f"[error] executable not found: {runner!r} (not on PATH)\n"
            f"Is {runner!r} installed in this environment?"
        )

    cwd = workspace
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
