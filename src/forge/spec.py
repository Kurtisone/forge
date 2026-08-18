"""
The delegation spec: what a job asks an implementer to do.

Everything here derives from _FIELDS, and that is the point. The
grammar, the prompt fragment listing the keys, the "what's still
missing" check, the question asked to fill a hole and the rendering
shown before a job is queued all read the same tuple. The router
already learned this the expensive way: its prompt and its grammar
were maintained separately, so a tool added to one and not the other
produced a model that named a tool the grammar forbade -- a 400 or a
fallback, depending on which side was stale.

The split of responsibility is deliberate and is the whole design:

    the grammar guarantees the SHAPE -- every key present, strings
    where strings belong, lists where lists belong

    the code decides COMPLETENESS -- which of those values are
    actually filled in

Asking the model to report whether its own spec is complete would be
asking it for a judgement, and every time that has been tried on this
repo it lost to a deterministic check (six times now, most recently
the escalation guard). An empty string is a fact about a string. It
does not need a 9B to notice it.
"""

import json
import re
from dataclasses import dataclass, field

from forge import gbnf
from forge.errors import SpecParseError
from forge.logger import log


@dataclass(frozen=True)
class Field:
    """
    One field of a spec.

    `question` is what gets asked when the field is empty, and it is
    written in French because it is shown to the user verbatim -- the
    same reason recall's answers had to be brought back to French
    after they drifted to English. `kind` drives the grammar branch.
    """

    name: str
    kind: str  # "text" | "list"
    required: bool
    question: str
    label: str
    # "restate" -- the draft may fill this, because a correct value is
    # a restatement of something in the request.
    # "ask" -- always asked, whatever the draft said, because a value
    # here is a JUDGEMENT about the task rather than a restatement of
    # it. See ground().
    source: str = "restate"


# Ordering is load-bearing twice over: it fixes the order of keys in
# the grammar (a fixed-order object is a far simpler GBNF rule than a
# free-order one, and simple is what survives contact with a 9B), and
# it fixes the order questions get asked in. Objective first, because
# every later question reads better once it exists.
_FIELDS: tuple[Field, ...] = (
    Field(
        name="objective",
        kind="text",
        required=True,
        label="Objectif",
        question="Qu'est-ce qui doit être fait, en une phrase ?",
    ),
    Field(
        name="workspace",
        kind="text",
        required=True,
        label="Emplacement",
        question="Dans quel dépôt ou quel dossier ?",
    ),
    Field(
        name="acceptance",
        kind="list",
        required=True,
        source="ask",
        label="Critères d'acceptation",
        question="À quoi verras-tu que c'est fait ? (un critère vérifiable par point)",
    ),
    Field(
        name="constraints",
        kind="list",
        required=False,
        source="ask",
        label="Contraintes",
        question="Y a-t-il des contraintes à respecter ? (« aucune » si non)",
    ),
    Field(
        name="context",
        kind="text",
        required=False,
        source="ask",
        label="Contexte",
        question="Un contexte utile à connaître ? (« aucun » si non)",
    ),
)

FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in _FIELDS)
_BY_NAME = {f.name: f for f in _FIELDS}

#: The only fields the model is allowed to draft, derived from the
#: field table rather than repeated here -- the same single-source
#: rule the rest of this module runs on.
#:
#: Enforced by the GRAMMAR rather than by filtering afterwards: the
#: draft call cannot emit a key outside this tuple, so for those
#: fields there is no invented value to detect in the first place.
DRAFTABLE: tuple[str, ...] = tuple(f.name for f in _FIELDS if f.source == "restate")


@dataclass
class Spec:
    """
    A delegation spec, complete or not.

    Mutable and default-empty on purpose: a spec is filled in over
    several turns as the user answers questions, so "half a spec" is a
    normal state to be in rather than an error to reject.
    """

    objective: str = ""
    workspace: str = ""
    acceptance: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    context: str = ""

    def value(self, name: str) -> object:
        return getattr(self, name)

    def set(self, name: str, value: object) -> None:
        f = _BY_NAME[name]
        if f.kind == "list":
            if isinstance(value, str):
                value = [line.strip() for line in value.splitlines() if line.strip()]
            setattr(self, name, [str(v).strip() for v in value if str(v).strip()])
        else:
            setattr(self, name, str(value).strip())

    def to_dict(self) -> dict:
        return {f.name: self.value(f.name) for f in _FIELDS}


def _check_field_names() -> None:
    """
    Field names have to survive being a JSON key inside a GBNF string
    literal AND being a dataclass attribute. Lowercase ASCII words
    satisfy both with room to spare, and the check is cheap next to
    the failure it prevents.
    """
    for f in _FIELDS:
        if not f.name.replace("_", "").isalnum() or not f.name.isascii():
            raise gbnf.GrammarError(
                f"field name {f.name!r} is not safe as a JSON key in a "
                "GBNF string literal"
            )


def field(name: str) -> Field:
    """The field definition behind a name."""
    return _BY_NAME[name]


def build_spec_grammar(names: tuple[str, ...] | None = None) -> str:
    """
    A GBNF grammar admitting exactly one spec object, keys in _FIELDS
    order.

    Field names land in the grammar as string LITERALS (the JSON keys)
    and never as rule names, so llama.cpp's no-underscore lexer rule
    does not reach them -- `acceptance_criteria` would be a perfectly
    legal key. Worth stating because the opposite is true one layer up
    in router/grammar.py, where tool names do become rule names, and
    assuming the same constraint applies here would be a plausible
    wrong guess. What a field name CAN break is the literal itself: a
    quote or a backslash in one closes the string early and produces a
    grammar llama-server rejects outright, which is a 400 on every
    completion rather than a bad spec. Hence _check_field_names.

    The rule names this function does emit (strlist, string, schar,
    hex, ws) are hyphen-free by construction, and validate() below is
    what keeps that true if one is ever added.

    Every key is REQUIRED by the grammar even when the field is
    optional, and an empty value is legal. That is what keeps the
    shape/completeness split clean: the model cannot omit a key (which
    would be a parse problem), it can only leave one empty (which is a
    question to ask).
    """
    _check_field_names()

    fields = _FIELDS if names is None else tuple(_BY_NAME[n] for n in names)

    members = []
    for f in fields:
        value_rule = "strlist" if f.kind == "list" else "string"
        members.append(f'"\\"{f.name}\\"" ws ":" ws {value_rule}')

    root = 'root ::= ws "{" ws ' + ' ws "," ws '.join(members) + ' ws "}" ws'
    rules = [
        root,
        r'string  ::= "\"" schar* "\""',
        r'schar   ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)',
        r"hex     ::= [0-9a-fA-F]",
        r"ws      ::= [ \t\n]*",
    ]
    # Only when a list field is actually in play: an unreferenced rule
    # is dead weight in a grammar llama.cpp parses on every call.
    if any(f.kind == "list" for f in fields):
        rules.insert(1, r'strlist ::= "[" ws (string (ws "," ws string)*)? ws "]"')
    grammar = "\n".join(rules) + "\n"

    # Validated here rather than trusted: this grammar is generated
    # from a tuple a future field will be appended to, and a name with
    # an underscore in it fails at llama-server as a 400 on every
    # completion, not as a bad spec.
    gbnf.validate(grammar)
    return grammar


def prompt_fields(names: tuple[str, ...] | None = None) -> str:
    """The field list as shown to the model, from the same tuple."""
    fields = _FIELDS if names is None else tuple(_BY_NAME[n] for n in names)
    lines = []
    for f in fields:
        shape = "liste de chaînes" if f.kind == "list" else "chaîne"
        state = "obligatoire" if f.required else "facultatif"
        lines.append(f'- "{f.name}" ({shape}, {state}) : {f.label}')
    return "\n".join(lines)


def parse(raw: str) -> Spec:
    """
    Read a spec out of a model answer.

    Under the grammar this is a plain json.loads. It is not written
    that way because the grammar only exists on llama.cpp: on ollama
    or OpenRouter the same call runs unconstrained (call_llm logs it),
    and the answer arrives wrapped in a fence or trailed by a
    sentence. Unknown keys are dropped rather than raising -- a model
    inventing a sixth field is not a reason to lose the five real
    ones.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise SpecParseError("no JSON object in the spec answer")

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise SpecParseError(f"spec answer is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise SpecParseError("spec answer is not a JSON object")

    spec = Spec()
    for name in FIELD_NAMES:
        if name in data and data[name] is not None:
            spec.set(name, data[name])
    return spec


_FRAGMENT_RE = re.compile(r"[a-z0-9]{3,}")


def ground(drafted: Spec, request: str) -> Spec:
    """
    Strip from *drafted* everything the request does not support.

    Written after the first real run, where two jobs out of three
    skipped the interview entirely because the model had filled every
    required field with plausible inventions -- a workspace nobody
    named and acceptance criteria nobody agreed to. The prompt says
    not to. It obeyed on vague requests ("corriger le cache KV") and
    ignored the instruction on specific ones ("migrer les tests vers
    pytest-asyncio"): the more the request gives it to work with, the
    more it extrapolates. Instructing a 9B not to do something is not
    a control, which is the same conclusion this repo has now reached
    seven times.

    The rule is that a draft may RESTATE but never CONCLUDE:

    - objective and workspace restate something in the request, so
      they survive if the request actually contains them. A workspace
      is kept only when one of its path fragments appears in the
      request, which lets "dans le dossier tools" become
      "src/forge/tools" while dropping a path invented whole.
    - acceptance, constraints and context are judgements about what
      "done" means. They are dropped unconditionally and asked. That
      is a question the user has to own anyway: acceptance criteria
      are the only thing making the spec checkable, and criteria the
      user never saw are worse than none.

    The cost is the case the draft was built for -- a complete request
    in one sentence, answered with no questions. In the first eight
    real jobs that case never occurred, and the only times the model
    filled everything it was inventing.
    """
    fragments = set(_FRAGMENT_RE.findall(request.lower()))
    grounded = Spec()

    for f in _FIELDS:
        if f.source == "ask":
            continue
        value = drafted.value(f.name)
        if not value:
            continue
        if f.name == "workspace" and not (
            fragments & set(_FRAGMENT_RE.findall(str(value).lower()))
        ):
            log.info("delegate: dropping ungrounded workspace %r", value)
            continue
        grounded.set(f.name, value)

    return grounded


def missing(spec: Spec) -> list[str]:
    """
    Required fields with nothing in them, in _FIELDS order.

    Deliberately not "fields the model thinks are underspecified": a
    vague objective is still an objective, and a spec good enough to
    argue about is worth more than a fourth round of questions. The
    user reviewing the rendered spec is the judgement step; this is
    only the emptiness check.
    """
    return [f.name for f in _FIELDS if f.required and not spec.value(f.name)]


def next_question(spec: Spec) -> tuple[str, str] | None:
    """
    The (field name, question) for the first hole, or None.

    One at a time, on purpose. A 9B asked three questions in one turn
    gets one answer covering one and a half of them, and then there is
    no deterministic way to know which field the answer belongs to --
    which is exactly the ambiguity awaiting_user has to avoid.
    """
    for name in missing(spec):
        return name, _BY_NAME[name].question
    return None


def render(spec: Spec) -> str:
    """
    The spec as shown in the thread before anything is queued.

    Empty optional fields are printed as "—" rather than hidden: the
    user is being asked to approve this, and a field that silently
    disappears cannot be noticed as wrong.
    """
    lines = []
    for f in _FIELDS:
        value = spec.value(f.name)
        if f.kind == "list":
            body = "\n" + "\n".join(f"  - {v}" for v in value) if value else " —"
        else:
            body = f" {value}" if value else " —"
        lines.append(f"**{f.label}** :{body}")
    return "\n".join(lines)
