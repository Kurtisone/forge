"""
What talking to this host means, defined once.

WHY THIS MODULE EXISTS

Three modules run commands against the machine -- `graphs/sysadmin.py`
and the two collectors in `harnais/collectors/` -- and until now only
one of them owned the plumbing. The other two reached into it:

    from forge.graphs.sysadmin import NO_OUTPUT, _collect_cmd, _run_fixed

Private names across a package boundary, and the collectors' own
docstrings say why it was done that way rather than duplicating:
rebuilding `podman ps` in a second place would also duplicate the
SYSADMIN_PODMAN_URL wiring, the minimal subprocess env, the timeout and
the "[error] " convention -- four things to drift instead of one.
`running_containers()` names the same cost: "two readings of `podman ps`
would drift the day the command grows a flag."

Both docstrings then name this module as the fix and say it was NOT
done, for one stated reason: `sysadmin` had to keep working untouched
while the Context Builder was built beside it. That constraint ended
with PR #79, which wired the Context Builder into the graph. So this is
the deferred half, done now that deferring it costs something.

It moves code and changes no behaviour. Every function below is the
body that was in `graphs/sysadmin.py`, and the graph imports them back
under the names it used, so its own callers and the forty-eight tests
that patch `sysadmin_mod._run_fixed` see exactly what they saw before.

WHAT THE MOVE ACTUALLY FIXED

One thing, and it is the duplication the collectors' docstrings warned
about, already present before this branch: the "--url when configured"
wiring for podman existed in THREE places -- `_DISCOVER_CONTAINERS_CMD`,
`_collect_cmd("container", ...)` and `collectors/containers._containers_cmd`
-- because each needed different arguments after it and there was no
seam to pass them through. `podman_cmd(*args)` is that seam, and the
same for `journalctl_cmd`. The proxy wiring is now stated once per tool.

WHERE IT LIVES, AND WHY NOT IN `graphs/`

The design document defines the Harnais as "la seule couche qui touche
la machine réelle" (docs/harnais.md, section 1). A graph asking the
Harnais how to reach the host is that sentence; the Harnais importing
from a graph was its inverse, and it is what the collectors had to write
to avoid duplicating. The dependency now points the way the document
says it does.
"""

import subprocess

from forge.config import (
    SYSADMIN_DBUS_ADDRESS,
    SYSADMIN_JOURNAL_DIR,
    SYSADMIN_MAX_LOG_LINES,
    SYSADMIN_PODMAN_URL,
)

#: What a command that ran fine and printed nothing is reported as.
#:
#: NOT "", and the difference has already cost a production bug: the
#: `nothing_collected` node tested `not collected.strip()` against this
#: value and could therefore never fire, so an empty journal reached the
#: model on every run -- while bench/sysadmin_verdict.py had measured
#: that this model, shown an empty log block, says it answers the
#: question. Anything comparing against "empty" must compare against
#: THIS, which is why it is one definition and not three.
NO_OUTPUT = "[no output]"


def podman_cmd(*args: str) -> list[str]:
    """`podman`, pointed at the read-only proxy when one is configured.

    Empty SYSADMIN_PODMAN_URL (the default, and every test that does
    not say otherwise) means "unchanged": no flag added, the exact
    command as before this was made configurable.
    """
    base = ["podman"]
    if SYSADMIN_PODMAN_URL:
        base += ["--url", SYSADMIN_PODMAN_URL]
    return base + list(args)


def journalctl_cmd(*args: str) -> list[str]:
    """`journalctl`, pointed at the bind-mounted journal when configured.

    Same "empty means unchanged" rule as podman_cmd above.
    """
    cmd = ["journalctl"]
    if SYSADMIN_JOURNAL_DIR:
        cmd += ["-D", SYSADMIN_JOURNAL_DIR]
    return cmd + list(args)


# Fixed, parameter-free discovery commands. Functions, not static
# lists: SYSADMIN_DBUS_ADDRESS/SYSADMIN_PODMAN_URL let these target a
# filtered proxy instead of the raw host bus/socket -- see config.py's
# comment above these three env vars, and deploy/README.md for the
# proxies themselves.
def discover_units_cmd() -> list[str]:
    # `systemctl list-units` was the original approach but had to be
    # abandoned: confirmed in production (SYSTEMD_LOG_LEVEL=debug)
    # that systemctl hardcodes a connection attempt at
    # /run/systemd/private first -- a systemd-specific shortcut
    # protocol, NOT standard D-Bus -- and never falls back to
    # DBUS_SYSTEM_BUS_ADDRESS (or any other address) if that exact
    # path is unavailable, which it always is inside a container whose
    # PID 1 isn't systemd. `busctl` has no such quirk: it speaks
    # standard D-Bus and honors --address correctly, confirmed
    # repeatedly against the same filtered proxy that systemctl
    # refused to use. --json=short gives a real parseable structure
    # (see graphs.sysadmin._parse_busctl_units) instead of the
    # columnar text `systemctl list-units` produces.
    cmd = ["busctl", "--json=short"]
    if SYSADMIN_DBUS_ADDRESS:
        cmd.append(f"--address={SYSADMIN_DBUS_ADDRESS}")
    return cmd + [
        "call",
        "org.freedesktop.systemd1",
        "/org/freedesktop/systemd1",
        "org.freedesktop.systemd1.Manager",
        "ListUnits",
    ]


def discover_containers_cmd() -> list[str]:
    """Names only. `harnais.collectors.containers` asks for more fields
    from the same `podman ps`, which is why the format string is an
    argument here rather than baked into podman_cmd."""
    return podman_cmd("ps", "--format", "{{.Names}}")


def collect_cmd(kind: str, name: str) -> list[str]:
    """Build a collection command. {name} is substituted only after
    collect_node has verified it against discover_node's own output --
    see collect_node's docstring. `kind` selects journalctl-by-unit,
    podman-logs, or journalctl-kernel; the journal dir / podman URL
    flags are added only when the matching proxy is configured.

    The name is passed as `--unit=<name>` and after a `--` separator
    respectively (audit M-4), never as a bare argument following a
    short flag. `-u foo` and `foo` are both positions where a value
    starting with `-` is read as an option instead: `-u --output=cat`
    or a container literally named `--help` would be interpreted by
    journalctl/podman rather than treated as a name. The attached form
    and the end-of-options marker remove that reading entirely.

    This is the second lock on a door that collect_node already
    bolted: a name only reaches here if it appeared verbatim in
    discovery output, and unit/container names don't normally start
    with a dash. What it defends is the case where discovery output is
    no longer trustworthy -- a hostile container name, or a proxy
    returning something the host didn't say -- which is exactly the
    assumption the validation rests on and therefore the one worth not
    resting the whole thing on.
    """
    if kind == "unit":
        return journalctl_cmd(
            f"--unit={name}", "--no-pager", "-n", str(SYSADMIN_MAX_LOG_LINES)
        )
    if kind == "container":
        return podman_cmd("logs", "--tail", str(SYSADMIN_MAX_LOG_LINES), "--", name)
    if kind == "kernel":
        return journalctl_cmd("-k", "--no-pager", "-n", str(SYSADMIN_MAX_LOG_LINES))
    raise ValueError(f"unknown collect kind: {kind!r}")


def subprocess_env() -> dict[str, str]:
    """Same minimal-env posture as tools/shell.py: no host env
    variables reach the subprocess except what's explicitly listed.
    DBUS_SYSTEM_BUS_ADDRESS is added only when SYSADMIN_DBUS_ADDRESS
    is configured, pointing busctl at the filtered proxy socket
    from deploy/forge-dbus-proxy.sh -- never the real system bus."""
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "TERM": "dumb"}
    if SYSADMIN_DBUS_ADDRESS:
        env["DBUS_SYSTEM_BUS_ADDRESS"] = SYSADMIN_DBUS_ADDRESS
    return env


def run_fixed(cmd: list[str], timeout: int) -> str:
    """Run a command whose every element is either a fixed literal or
    a name already verified against discover_node's own output.
    Never shell=True, never a hand-built string -- same posture as
    tools/shell.py's allowlisted subprocess.run(parts, ...). Uses
    subprocess_env() so DBUS_SYSTEM_BUS_ADDRESS (when configured)
    points busctl at the filtered proxy, not the host bus."""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=subprocess_env(),
        )
    except FileNotFoundError:
        return f"[error] executable not found: {cmd[0]!r}"
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except OSError as e:
        return f"[error] OS error: {e}"

    output = (result.stdout or result.stderr or "").strip()
    lines = output.splitlines()[:SYSADMIN_MAX_LOG_LINES]
    joined = "\n".join(lines) if lines else NO_OUTPUT

    if result.returncode != 0:
        # The command RAN (no Python-level exception above) but the
        # target itself failed -- e.g. busctl unable to reach the bus,
        # podman unable to reach its socket. This must carry the same
        # "[error]" prefix as the exception-based cases above:
        # without it, a real production case slipped straight through
        # as if it were valid data. The case that taught this was
        # systemctl, back when discovery still used it: its two-line
        # failure message ("System has not been booted with
        # systemd...\nFailed to connect to bus...") got parsed as two
        # fake unit names ("System", "Failed") by _discover_node, and
        # podman's connection-refused text got parsed as a fake
        # container name the same way. Caught in production on
        # 2026-08-11. The systemctl path is gone; the failure mode it
        # exposed is not, which is why the guard stays.
        return f"[error] {cmd[0]} exited {result.returncode}: {joined}"

    return joined
