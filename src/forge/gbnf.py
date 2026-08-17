"""
Shared GBNF checks.

Until v3.13 there was exactly one grammar in Forge (the router's), so
the rules it has to satisfy lived in tests/test_grammar.py. The
delegation spec adds a second one, and the checks are not
grammar-specific -- they encode how llama.cpp's own grammar lexer
reads a file, not anything about routing. Keeping them in a test
would mean the next grammar silently gets none of them.

The check that matters most is `check_rule_names`. llama.cpp builds
rule names out of is_word_char(), which accepts [a-zA-Z0-9-] and NOT
underscore, so a name like `payload_call` lexes as the rule `payload`
followed by garbage. llama-server then rejects the WHOLE grammar with
"expecting newline or end at _call" and answers 400 to every
completion -- the router doesn't degrade, it dies. That cost a full
debugging cycle in v3.10, and the HTTP body llama-server returns says
only "failed to parse grammar": it never names the offending rule.
Validating here is how the rule name ends up in a Forge log instead.
"""

import re

from forge.errors import ForgeError

# A rule definition: a name at the start of a line, then ::=. The
# name pattern is deliberately PERMISSIVE (underscores included) --
# this is what finds the names llama.cpp would reject, so it has to
# be able to match them in the first place.
_DEFINITION_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*::=", re.MULTILINE)
_STRING_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_CHAR_CLASS_RE = re.compile(r"\[\^?(?:[^\]\\]|\\.)*\]")
_IDENTIFIER_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")


class GrammarError(ForgeError):
    """A generated GBNF grammar llama.cpp would refuse to parse."""


def is_word_char(c: str) -> bool:
    """
    llama.cpp's own is_word_char(), from its grammar lexer.

    Reimplemented rather than approximated: it accepts [a-zA-Z0-9-]
    and NOT underscore, which is the single fact this function exists
    to pin down.
    """
    return c.isascii() and (c.isalpha() or c.isdigit() or c == "-")


def rule_names(grammar: str) -> set[str]:
    """Names defined by the grammar (left-hand side of a ::=)."""
    return set(_DEFINITION_RE.findall(grammar))


def referenced_names(grammar: str) -> set[str]:
    """
    Names referenced in rule bodies.

    String literals and character classes are stripped FIRST, so a
    JSON key spelled "\\"tool\\"" inside a rule body is not mistaken
    for a reference to a rule named `tool`. Doing it any other way
    needs a hand-maintained keyword list, which is one more thing to
    remember to update every time a rule is added.
    """
    body = _STRING_LITERAL_RE.sub(" ", grammar)
    body = _CHAR_CLASS_RE.sub(" ", body)
    return set(_IDENTIFIER_RE.findall(body))


def check_rule_names(grammar: str) -> list[str]:
    """Names containing a character llama.cpp's lexer would reject."""
    return sorted(
        name
        for name in rule_names(grammar) | referenced_names(grammar)
        if any(not is_word_char(c) for c in name)
    )


def check_references(grammar: str) -> list[str]:
    """Names referenced by a rule body but never defined."""
    return sorted(referenced_names(grammar) - rule_names(grammar))


def validate(grammar: str) -> None:
    """
    Raise GrammarError -- naming what is wrong -- if llama.cpp would
    refuse this grammar.

    Callers are expected to CATCH this rather than let it propagate:
    sending a grammar the server will reject guarantees a 400 on every
    completion, while dropping it degrades to the parser's existing
    fallback chain. A named error in the log plus a degraded run beats
    an unnamed 400 and no run at all.
    """
    if not grammar.strip():
        raise GrammarError("empty grammar")

    if "root" not in rule_names(grammar):
        raise GrammarError("grammar has no `root` rule")

    bad_names = check_rule_names(grammar)
    if bad_names:
        raise GrammarError(
            "rule names contain characters llama.cpp's lexer rejects "
            f"(only [a-zA-Z0-9-] is accepted, note the missing underscore): "
            f"{', '.join(bad_names)}"
        )

    undefined = check_references(grammar)
    if undefined:
        raise GrammarError(f"referenced but undefined rules: {', '.join(undefined)}")
