"""
Tool discovery and lookup.

A tool is any module in forge.tools that exposes a `run(content: str)
-> str` function. Loading is explicit and observable: a tool that
fails to import is logged as a warning, never swallowed silently
(the previous behavior made broken tools invisible).
"""

import importlib
import pkgutil

import forge.tools as tools_pkg
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

        if hasattr(mod, "run"):
            TOOLS[module.name] = mod.run
            log.event("tool.registered", name=module.name)
        else:
            log.debug("module %r has no run(), skipped", module.name)


def get_tool(name: str):
    return TOOLS.get(name)


def available_tools() -> list[str]:
    return sorted(TOOLS.keys())
