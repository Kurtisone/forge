"""
The D-Bus proxy's filter, pinned.

WHY THIS FILE EXISTS

`deploy/forge-dbus-proxy.sh` is the only thing between the `forge`
container and the host's systemd. Its whole security property is one
flag (`--filter`) and one list (the `--call` rules), and until this file
existed nothing checked either: the podman proxy's filter has
tests/test_podman_ro_proxy.py, and this one had none.

That is not a hypothetical gap. Dropping `--filter` does not break
anything visible -- `ListUnits` still works, discovery still returns 317
units, sysadmin still answers -- and the proxy becomes a plain forwarder
to the bus. The failure looks exactly like success.

WHAT IS PINNED, AND WHY EACH ONE

  - `--filter` is present. Without it xdg-dbus-proxy forwards
    everything.
  - The ONLY options are `--filter` and `--call=`. `--talk`, `--own`,
    `--see` and `--broadcast` each widen what a client may reach, and
    none of them is needed to answer ListUnits.
  - Every `--call` names an object path (`...@/org/...`). A rule with no
    path allows the method on ANY object.
  - The set of `--call` rules is EXACTLY the reviewed one below. Widening
    the proxy is a decision, and the place it gets made is an edit to
    ALLOWED_CALLS in this file, in the diff, where a reviewer sees it.
  - The upstream is the SYSTEM bus. Pointing the same allowlist at the
    session bus is a different risk, not a different address: there the
    user owns the manager, so a mistake in the filter is arbitrary code
    execution as that user, where on the system bus polkit would refuse.
  - What the code asks systemd for is allowed. Tightening the list
    cannot silently blind sysadmin: the call `discover_units_cmd()`
    builds has to be one of the rules.
"""

import pathlib
import re
import shlex

from forge.harnais.host_exec import discover_units_cmd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "forge-dbus-proxy.sh"

SYSTEM_BUS = "unix:path=/run/dbus/system_bus_socket"
_MANAGER = "org.freedesktop.systemd1"
_PATH = "/org/freedesktop/systemd1"

#: Every call the proxy lets through, in the spelling xdg-dbus-proxy
#: takes: NAME=INTERFACE.MEMBER@PATH. Widening this is a decision; make
#: it here. It was five rules until 2026-09-26; the four removed
#: (ListUnitsByPatterns, GetUnit, Properties.Get, Properties.GetAll) were
#: never called, and GetAll on the manager returns its Environment.
ALLOWED_CALLS = {
    f"{_MANAGER}={_MANAGER}.Manager.ListUnits@{_PATH}",
}


def _proxy_args() -> list[str]:
    """The arguments of the `exec xdg-dbus-proxy ...` line, split the way
    the shell splits them, with backslash continuations joined first."""
    text = re.sub(r"\\\n\s*", " ", SCRIPT.read_text(encoding="utf-8"))
    lines = [
        ln for ln in text.splitlines() if ln.lstrip().startswith("exec xdg-dbus-proxy")
    ]
    assert len(lines) == 1, (
        "expected exactly one `exec xdg-dbus-proxy` line in "
        f"{SCRIPT.name}, found {len(lines)} -- did the script's shape change?"
    )
    return shlex.split(lines[0])[2:]


def _options() -> list[str]:
    return [a for a in _proxy_args() if a.startswith("--")]


def _positionals() -> list[str]:
    return [a for a in _proxy_args() if not a.startswith("--")]


def _calls() -> set[str]:
    return {o.removeprefix("--call=") for o in _options() if o.startswith("--call=")}


def test_the_filter_is_on():
    assert "--filter" in _options(), (
        "forge-dbus-proxy.sh has no --filter: xdg-dbus-proxy then forwards "
        "EVERYTHING, and nothing visible breaks when that happens"
    )


def test_the_only_options_are_filter_and_call():
    stray = [o for o in _options() if o != "--filter" and not o.startswith("--call=")]
    assert not stray, (
        f"{stray}: --talk/--own/--see/--broadcast each widen what the "
        "container can reach, and none is needed to answer ListUnits"
    )


def test_every_call_rule_names_an_object_path():
    """A rule without `@/path` allows the method on any object."""
    bare = sorted(c for c in _calls() if "@/" not in c)
    assert not bare, f"rules with no object path: {bare}"


def test_the_allowed_calls_are_exactly_the_reviewed_set():
    calls = _calls()
    assert calls == ALLOWED_CALLS, (
        f"added: {sorted(calls - ALLOWED_CALLS)}; removed: "
        f"{sorted(ALLOWED_CALLS - calls)}. Widening this proxy is a decision: "
        "edit ALLOWED_CALLS in this file, in the same diff."
    )


def test_the_script_listens_on_one_socket_in_front_of_the_system_bus():
    assert _positionals() == [SYSTEM_BUS, "$PROXY_SOCKET"], _positionals()


def test_what_the_code_asks_systemd_for_is_allowed():
    """
    The inverse guard: tightening the list must not blind sysadmin. The
    busctl call discover_units_cmd() builds, and the units collector
    reuses, has to be one of the rules.
    """
    cmd = discover_units_cmd()
    service, path, interface, member = cmd[cmd.index("call") + 1 :][:4]
    assert f"{service}={interface}.{member}@{path}" in _calls()
