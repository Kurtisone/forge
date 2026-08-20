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


class SpecParseError(ForgeError):
    """The model did not return a usable delegation spec."""


class ToolNotFoundError(ForgeError):
    """The router asked for a tool that is not registered."""


class ToolExecutionError(ForgeError):
    """A tool raised while executing."""


class LoopGuardError(ForgeError):
    """The orchestrator stopped to prevent an infinite / cyclic run."""


class CapabilityAmbiguousError(ForgeError):
    """
    Several providers answer for one capability and nothing is
    entitled to pick between them.

    Choosing is the Cognitive Scheduler's job (ARCHITECTURE.md,
    Niveau 3), which does not exist yet. Until it does, this is
    unreachable -- every capability has exactly one provider. It is
    typed now so the day it becomes reachable, it surfaces as a named
    architectural gap instead of dispatch quietly taking the first
    candidate and nobody noticing which one ran.
    """
