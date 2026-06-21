"""
Forge error hierarchy.

Every error that can happen inside the runtime is typed, so the
orchestrator never has to guess what failed by parsing a string.
"""


class ForgeError(Exception):
    """Base class for all Forge runtime errors."""


class ProviderError(ForgeError):
    """The LLM backend (ollama / llama.cpp / openrouter) failed to answer."""


class RouterParseError(ForgeError):
    """The router did not return a usable JSON instruction."""


class ToolNotFoundError(ForgeError):
    """The router asked for a tool that is not registered."""


class ToolExecutionError(ForgeError):
    """A tool raised while executing."""


class LoopGuardError(ForgeError):
    """The orchestrator stopped to prevent an infinite / cyclic run."""
