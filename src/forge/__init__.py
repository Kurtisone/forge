"""
Forge -- a lightweight LLM agent runtime.

The version lives here and nowhere else.

It used to live in two places: `version` in pyproject.toml and a
string literal in api.py's FastAPI() call. Both said "3.3.0" while the
working version was 3.13 -- ten release cycles of drift, on a number
that OpenAPI serves to anyone who reads /openapi.json. Neither copy
was wrong when it was written; nothing ever made them wrong out loud.

So: one literal, read by everyone. pyproject.toml declares the version
dynamic and points setuptools at this attribute, api.py imports it,
and tests/test_version.py fails if any of that stops being true.

Kept as a plain module-level literal on purpose -- setuptools reads
this file statically (it does not import it), so anything computed
would break the build backend rather than the runtime, which is a much
worse place to find out.
"""

__version__ = "3.13.0"
