"""
The Harnais -- the only layer that touches the real machine.

It never reasons, never sees the user's question, and does not know an
LLM exists (document section 2). It turns raw readings into
timestamped `Fact`s with a named source, and says plainly when it
could not read something.

The MVP is observation only. There is no ActionCapability here and
there is not meant to be one until V3: what exists today proposes and
a human applies, the same posture graphs/sysadmin.py and tools/git.py
already hold.
"""

from forge.harnais.facts import Correlation, Fact, Hypothesis

__all__ = ["Correlation", "Fact", "Hypothesis"]
