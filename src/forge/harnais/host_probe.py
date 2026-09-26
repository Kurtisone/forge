"""
Can Forge see the host at all? Answered where somebody already looks.

WHERE THIS CAME FROM

From 2026-09-11 to 2026-09-14 both host proxies sat in a restart loop
(the checkout had moved, python exited status=2) and for three days
sysadmin was blind: 0 containers, 0 units, every target missed. Nothing
said so. The Harnais now says it INSIDE an answer -- a failed
Observation reads "could not observe" -- but only to whoever asks
sysadmin a question, and nobody did until the third day. Forge is
reactive; a failure nobody asks about is a failure nobody sees.

This is the same fact, reported where somebody already looks: /health,
which the web UI's status line reads on load, after every turn and
whenever the tab comes back.

WHAT IT ASKS

The two collectors sysadmin itself runs and nothing else: the
containers collector for the podman proxy, the units collector for the
D-Bus one. Reusing them means there is one definition of "reachable"
and it is the one that matters -- /health cannot say ok about a path
sysadmin then fails on, or the reverse. It also means this adds no
access to the host: both sockets are already mounted into the container.

It asks only about a proxy that is CONFIGURED. With no
SYSADMIN_PODMAN_URL / SYSADMIN_DBUS_ADDRESS there is no proxy to be
blind through, and running a collector anyway interrogates whatever
machine happens to run this process -- the mistake that made four
refusal tests fail on CI, in tests/test_sysadmin_harnais_path.py.

THE STATES

"ok", "unobservable" (the collector reported a failure, for whatever
reason) and "unknown" (the very first probe has not finished). Never the
error text: /health is unauthenticated, so the reason goes to the log,
once, when the state changes -- which also makes the loss visible in
`podman logs forge` whether or not anyone is polling.

COST, AND WHY IT IS SHAPED THIS WAY

A probe spawns subprocesses, and /health is open and hit by the UI and
by the phone. Results are reused for _TTL_S and at most one refresh runs
at a time, so the work is bounded whatever the request rate. A caller
waits at most _WAIT_S for a refresh, because the API serves everything
from a pool of two workers and a hung proxy (each collector may take
SYSADMIN_DISCOVERY_TIMEOUT to give up) must not be able to hold both.
Past that wait it gets the previous answer, or "unknown" if there is
none.
"""

from __future__ import annotations

import threading
import time

from forge.config import ENABLED_TOOLS, SYSADMIN_DBUS_ADDRESS, SYSADMIN_PODMAN_URL
from forge.logger import log

#: How long an answer is reused. Short enough that a proxy lost between
#: two polls shows on the next one; long enough that the UI's own polling
#: (page load, every turn, every tab switch) costs one probe, not many.
_TTL_S = 30.0

#: The longest a /health caller waits for a refresh in progress.
_WAIT_S = 1.5

#: Indirection so a test can move time without touching the real clock,
#: which the threading module also reads.
_now = time.monotonic

_lock = threading.Lock()
_result: dict[str, str] | None = None
_at = 0.0
_thread: threading.Thread | None = None


def _configured() -> dict[str, type]:
    """proxy name -> the collector that observes through it, for each
    proxy this deployment actually routes through. Empty when sysadmin is
    off or neither proxy is configured."""
    if "sysadmin" not in ENABLED_TOOLS:
        return {}

    from forge.harnais.collectors.containers import ContainersCollector
    from forge.harnais.collectors.units import SystemUnitsCollector

    checks: dict[str, type] = {}
    if SYSADMIN_PODMAN_URL:
        checks["podman"] = ContainersCollector
    if SYSADMIN_DBUS_ADDRESS:
        checks["systemd"] = SystemUnitsCollector
    return checks


def _refresh(checks: dict[str, type]) -> None:
    """Run the collectors and record what they saw. Runs in its own
    thread, so it never raises: a probe that crashes must leave the last
    answer standing, not kill anything."""
    global _result, _at
    try:
        from forge import harnais

        names = list(checks)
        observations = harnais.observe([checks[n]() for n in names])
        seen = {
            n: "unobservable" if o.failed else "ok"
            for n, o in zip(names, observations, strict=True)
        }
        with _lock:
            before = _result
            _result, _at = seen, _now()

        for n, o in zip(names, observations, strict=True):
            was = (before or {}).get(n)
            if o.failed and was != "unobservable":
                log.warning(
                    "host access lost: %s is unobservable (%s)", n, o.error[:200]
                )
            elif not o.failed and was == "unobservable":
                log.info("host access back: %s is observable again", n)
    except Exception as e:  # noqa: BLE001 -- see docstring
        log.warning("host probe crashed: %s: %s", type(e).__name__, e)


def host_access() -> dict[str, str]:
    """
    {proxy: "ok" | "unobservable" | "unknown"} for each configured proxy,
    or {} when there is nothing to report.

    Blocking, so call it from a worker thread: on a stale answer it waits
    up to _WAIT_S for the refresh it starts or joins.
    """
    global _thread
    checks = _configured()
    if not checks:
        return {}

    with _lock:
        if _result is not None and _now() - _at < _TTL_S:
            return dict(_result)
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(
                target=_refresh, args=(checks,), name="host-probe", daemon=True
            )
            _thread.start()
        running = _thread

    running.join(_WAIT_S)

    with _lock:
        if _result is not None:
            return dict(_result)
    return dict.fromkeys(checks, "unknown")


def reset() -> None:
    """Forget the last answer. For tests: the state is module-level, and
    a test that left an answer behind would decide the next one."""
    global _result, _at, _thread
    with _lock:
        _result, _at, _thread = None, 0.0, None
