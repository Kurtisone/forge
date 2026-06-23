"""
Tool discovery and lookup.

A tool is any module in forge.tools that exposes a `run(content: str)
-> str` function -- but having run() is necessary, not sufficient.
A module is only ever dispatchable if its name is also listed in
ENABLED_TOOLS (config.py). This is deliberate: implementing run() in
a stub like shell.py must not silently make it reachable from router
output the moment someone fills it in. Loading is explicit and
observable either way: a tool that fails to import, or that exists
but isn't enabled, is logged as a warning, never swallowed silently.
"""

import importlib
import pkgutil

import forge.tools as tools_pkg
from forge.config import ENABLED_TOOLS
from forge.logger import log

TOOLS = {}

_RESERVED_MODULES = {"registry"}


def load_tools() -> None:
    TOOLS.clear()

    for module in pkgutil.iter_modules(tools_pkg.__path__):
        if module.name in _RESERVED_MODULES:
            continue

        try:
            mod = importlib.import_module(f"forge.tools.{module.name}")
        except Exception as e:  # noqa: BLE001
            log.warning("tool %r failed to import: %s", module.name, e)
            continue

        if not hasattr(mod, "run"):
            log.debug("module %r has no run(), skipped", module.name)
            continue

        if module.name not in ENABLED_TOOLS:
            log.warning(
                "tool %r has a run() handler but is not in ENABLED_TOOLS, "
                "skipping (set ENABLED_TOOLS to opt in)",
                module.name,
            )
            continue

        TOOLS[module.name] = mod.run
        log.event("tool.registered", name=module.name)


def get_tool(name: str):
    return TOOLS.get(name)


def available_tools() -> list[str]:
    return sorted(TOOLS.keys())
