"""
Single, central logging surface for Forge.

Rule: nothing else in the codebase calls print(). Every module that
wants to observe what's happening imports `log` from here. This is
the "logs" leg of the LLM / tools / logs separation -- it is
read-only with respect to the other two: it never influences control
flow, it only records it.

Debug mode is controlled by SHOW_DEBUG (env var), not hardcoded.
"""

import logging
import sys

from forge.config import SHOW_DEBUG

_LEVEL = logging.DEBUG if SHOW_DEBUG else logging.INFO

_logger = logging.getLogger("forge")
_logger.setLevel(_LEVEL)

if not _logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    _logger.addHandler(handler)
    _logger.propagate = False


class _Log:
    """Thin wrapper so call sites read as log.debug(...) / log.event(...)."""

    def debug(self, msg: str, *args) -> None:
        _logger.debug(msg, *args)

    def info(self, msg: str, *args) -> None:
        _logger.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        _logger.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        _logger.error(msg, *args)

    def event(self, event_name: str, **fields) -> None:
        """
        Structured trace point, only emitted in debug mode.
        Used for router prompt/output, tool dispatch, latencies, etc.
        """
        if not SHOW_DEBUG:
            return
        payload = " ".join(f"{k}={fields[k]!r}" for k in fields)
        _logger.debug("%s %s", event_name, payload)


log = _Log()
