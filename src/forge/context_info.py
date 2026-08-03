"""
Small shared context fact injected into LLM prompts.

The model has no reliable notion of "today" on its own -- only a
stale sense of the current date from training, which produced a
visibly wrong result in production (confusing 2025 and 2026 in a
research synthesis) until the user spelled the date out explicitly
in their own prompt. Centralized here rather than duplicated at each
call site (router/prompt.py, graphs/review.py, graphs/research.py)
after the same kind of drift already happened once with duplicated
cleaning logic -- see text_cleaning.py's module docstring.
"""

from datetime import date


def today_line() -> str:
    """A single prompt line stating today's date. ISO format --
    locale-independent and unambiguous, no dependency on the
    container having a French (or any) locale installed.

    Deliberately the machine's local naive date, not timezone-aware
    UTC: Forge runs as a single instance on the user's own machine
    (NiPoGi), so "today" means today where that machine physically
    is -- a UTC date would be wrong for roughly half of every day
    relative to the user's actual local evening/night.
    """
    return f"Today's date is {date.today().isoformat()}."  # noqa: DTZ011
