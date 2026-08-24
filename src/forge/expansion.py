"""
More ways of asking the same question.

WHY

Measured against the real store on 2026-08-24, under the query
instruction that ships by default:

    "Quel processeur a mon NiPoGi ?"      #308 at rank 1, 0.7289
    "Tu peux me lister mon matériel ?"    #308 nowhere in the top 5

Same information, one phrasing that finds it and one that does not.
No value of RECALL_MAX_DISTANCE fixes that -- the entry never comes
back to be filtered. The 2026-08-22 measurement said the same thing
from the other end: a fact reading `Matériel : NiPoGi AM06PRO,
processeur Ryzen 5500U, 32 Go de RAM` sat at rank 109 for "Tu peux me
lister mon matériel ?", with the word "matériel" in it.

WHY THE QUERY AND NOT THE ENTRY

Decided on 2026-08-22 and it still holds: extend the QUERY at read
time, never the FACT at write time. A wrong expansion of a query
searches somewhere useless and the distance cutoff throws the result
away. A wrong expansion of a fact writes something nobody said into
memory, where it is indistinguishable from something they did, and it
stays.

TWO LAYERS, MEASURED SEPARATELY

`terms` is this module: deterministic, free, no model. It strips the
conversational frame a spoken question carries and offers a
content-word form. It fixes phrasing overhead and it does NOT bridge
vocabulary -- nothing deterministic turns "matériel" into
"processeur". Said out loud here because a mechanism that sounds like
it should help is exactly the kind that gets credited for a result it
did not produce; bench/recall_expansion.py scores the two modes in
separate columns for that reason.

`llm` is the half that can bridge vocabulary, at the price of one
model call. MEASURED ON THE REAL STORE, 2026-08-24: it moved #308
from absent-from-the-top-5 to rank 2, on the exact question that
motivated this module.

AND `terms` LOST, on every question it was asked -- see its docstring
for the four numbers. It is kept because a measured negative is worth
being able to reproduce, and because it is free to re-run if the
embedding model ever changes. It is no longer composed into `llm`,
and it should not be turned on.
"""

import json
import re
import unicodedata

from forge.errors import ProviderError
from forge.llm import call_llm
from forge.logger import log

#: The modes RECALL_EXPANSION accepts.
MODES = ("off", "terms", "llm")

# Ceiling on how many EXTRA queries leave this module, whatever the
# mode. Each one costs an embedding call on the rescue path, and past
# a handful the merge stops being "the same question asked again" and
# becomes a trawl -- more rows from further away, which is what the
# cutoff exists to refuse.
MAX_VARIANTS = 4

# A variant this long has stopped being a query. Not a hard failure,
# just dropped: the model that produced it has usually answered the
# question instead of rephrasing it.
_MAX_VARIANT_CHARS = 120

# The leading frame a question carries when it is spoken to someone
# rather than typed into a search box. Stripped as a whole, not word
# by word, so "Tu peux me lister mon matériel ?" becomes "lister mon
# matériel" and not a bag of fragments.
_FRAME_RE = re.compile(
    r"^\s*(?:est-ce\s+que\s+)?"
    r"(?:tu\s+peux|peux[-\s]tu|tu\s+pourrais|pourrais[-\s]tu|"
    r"tu\s+sais|sais[-\s]tu|dis[-\s]moi|rappelle[-\s]moi|"
    r"j['\u2019]aimerais\s+savoir|je\s+voudrais\s+savoir|"
    r"je\s+cherche|montre[-\s]moi|donne[-\s]moi)"
    r"\s+(?:me\s+|m['\u2019]|te\s+)?",
    re.IGNORECASE,
)

# Trailing politeness and punctuation. "?" is not noise to a reader,
# but to an embedding it is one more token the stored fact does not
# have.
_TAIL_RE = re.compile(
    r"[\s,]*(?:s['\u2019]il\s+te\s+pla[iî]t|s['\u2019]il\s+vous\s+pla[iî]t|stp|svp|merci)?"
    r"[\s?!.\u2026]*$",
    re.IGNORECASE,
)

# Deliberately SMALL, and French-first because that is what this
# deployment is asked in. Only words that carry no retrieval signal at
# all: articles, possessives, pronouns, the interrogatives, and the
# auxiliaries. Every word not on this list survives, which is the safe
# direction -- dropping a content word loses the query, keeping a
# stopword costs almost nothing.
# Written as prose and split, rather than as a literal collection: a
# 90-element set one word per line is unreadable, and the point of
# keeping this list small is that a reader can check it.
_STOPWORD_TEXT = """
a à ai as au aux avec avoir c ça ce ces cet cette combien comment d dans
de des du elle en est et être eu il ils j je l la le les leur lui m ma
mais me mes moi mon n ne nos notre nous on ont ou où par pas peux pour
pourquoi qu quand que quel quelle quelles quels qui quoi sa sais se ses
son sont sur t ta te tes toi ton tu un une vos votre vous y
a-t-il est-ce peux-tu sais-tu
a and are do does for is it me my of the what which
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# A one-word query matches everything and nothing -- the same argument
# rag._MIN_ENTRY_WORDS makes about a one-word entry, from the other
# side of the search.
_MIN_VARIANT_WORDS = 2


def _fold(text: str) -> str:
    """
    Lowercase, accent-free, punctuation-free -- for COMPARING two
    strings, never for searching with them.

    Only ever used to decide whether a variant is the query again in
    different clothes. Embedding the folded form instead would strip
    the accents off a French store's own vocabulary, which is the
    opposite of the point.
    """
    stripped = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in stripped if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[\w-]+", stripped))


def _without_frame(query: str) -> str:
    """The question with its conversational scaffolding removed."""
    return _TAIL_RE.sub("", _FRAME_RE.sub("", query)).strip()


def _content_words(query: str) -> str:
    """
    The question reduced to the words a stored note would share with
    it.

    Tokens keep their accents, their hyphens and their case:
    `sqlite-vec`, `NiPoGi` and `Q4_K_M` are the words that make a hit
    here, and a tokenizer that splits or lowercases them is throwing
    away the only vocabulary this store has that nothing else does.
    """
    words = [
        word
        for word in re.findall(r"[^\W_]+(?:[-'\u2019_][^\W_]+)*", query, re.UNICODE)
        if _fold(word) not in _STOPWORDS
    ]
    return " ".join(words)


def _acceptable(variant: str, query: str, seen: set[str]) -> bool:
    folded = _fold(variant)
    if not folded or folded in seen:
        return False
    if folded == _fold(query):
        return False
    # Counted on the variant itself, not on its folded form: folding
    # splits `m'appelle` into two tokens, so a variant that is visibly
    # one word passed this check and reached the store as a query
    # matching everything.
    if len(variant.split()) < _MIN_VARIANT_WORDS:
        return False
    return len(variant) <= _MAX_VARIANT_CHARS


def keep(candidates: list[str], query: str) -> list[str]:
    """
    The candidates worth spending an embedding call on, in order.

    Dropped: anything that folds to the query itself (a variant that
    is the question again buys nothing but latency), anything already
    proposed, anything under two words, and anything long enough to be
    an answer rather than a query. Shared by both modes so that a
    model-written variant passes exactly the checks a deterministic
    one does.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if _acceptable(candidate, query, seen):
            seen.add(_fold(candidate))
            kept.append(candidate)
    return kept[:MAX_VARIANTS]


def terms(query: str) -> list[str]:
    """
    The deterministic variants: unframed, then content words only.

    MEASURED WORSE, and kept only so that stays reproducible. Against
    the real store on 2026-08-24, distance to the named entry (or to
    the closest row where it was absent):

        question                            baseline    terms
        Tu peux me lister mon matériel ?      0.9083   1.0277
        Combien de RAM a le NiPoGi ?          0.7336   0.8591
        Comment s'appelle mon chat ?          0.9495   1.0821
        Quelle est la recette … tatin ?       1.1578   1.1609

    Four out of four, in the same direction, hits and misses alike.
    The embedding model is instruction-tuned on natural-language
    queries; a keyword bag is off-distribution for it, and even the
    mild rewrite ("lister mon matériel", still a phrase) lost by more
    than a tenth. Stripping words from a question does not make it a
    better query here -- it makes it a worse sentence.

    Which is the useful shape of the finding: what worked was the
    model's rewrites, and those are PHRASES in the store's own
    vocabulary, not the question with its function words removed.

    No model, no network, no configuration. Either can come back
    identical to the query or to each other, in which case `keep`
    drops it and the caller pays nothing.

    The second is derived from the FIRST and not from the raw query:
    the frame is a phrase, not a bag of stopwords, and "s'il te
    plaît" survives a stopword filter as `plaît` -- a word no stored
    fact has ever contained.
    """
    return keep(_term_candidates(query), query)


def _term_candidates(query: str) -> list[str]:
    """The two rewrites, before `keep` has had an opinion on them."""
    unframed = _without_frame(query)
    return [unframed, _content_words(unframed)]


def variants(query: str, mode: str) -> list[str]:
    """
    The extra queries to search alongside the original, for a mode.

    Unknown modes yield nothing and say so. Falling back to the full
    behaviour on a typo would turn `RECALL_EXPANSION=trems` into a
    model call nobody asked for; falling back to none of it is the
    direction where a mistake costs what it did before this module
    existed.
    """
    query = (query or "").strip()
    if not query or mode == "off":
        return []
    if mode == "terms":
        return terms(query)
    if mode == "llm":
        # The model's rewrites ALONE. `terms` used to be appended here
        # on the theory that a free variant costs nothing -- measured
        # on 2026-08-24, it costs two things. It pushes distances up
        # on this embedding model (four questions out of four), and
        # its rows compete for the merge's top_k slots, so a variant
        # that finds nothing useful can still push the rescued entry
        # out of the list.
        #
        # A failed call therefore yields no variants and no rescue,
        # rather than a rescue built out of the rewrites that lost.
        # Unhelped, never wrong.
        return keep(_from_llm(query), query)
    log.warning(
        "unknown RECALL_EXPANSION=%r, expansion is off (expected one of: %s)",
        mode,
        ", ".join(MODES),
    )
    return []


# Exactly three strings, fixed shape. Not `{1,3}` and not a free
# array: a fixed-arity rule is the simplest thing a 9B can be held to,
# which is the same reason spec.py fixes its key order rather than
# allowing any order. What comes back is then trimmed by `keep`, so
# "three" is a sampling constraint and not a promise about the output.
#
# Rule names are hyphen-free by construction here, but gbnf.validate
# is still what checks it: llama.cpp's lexer rejects underscores in
# rule names and answers 400 to EVERY completion when it does, which
# is a dead router rather than a degraded one (v3.10, ffa9542).
_GRAMMAR = r"""root ::= ws "[" ws string ws "," ws string ws "," ws string ws "]" ws
string ::= "\"" schar* "\""
schar ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
"""

# The example's subject, which must never appear in a variant that
# came back. Same net as recall's _EXAMPLE_LEAK_FRAGMENTS and for the
# same measured reason: this model copies a worked example verbatim
# when the real input is unfamiliar. The tarte tatin is this
# repository's own standing example of a question the store cannot
# answer (bench/recall_distance.py), so a variant about it is
# recognisable AND harmless to drop.
_EXAMPLE_SUBJECT = "tatin"

_PROMPT = """/no_think
The question below will be matched against short notes stored in a
personal memory. Retrieval there follows the WORDS as much as the
meaning: a note is found when it is written with the words the query
uses. Measured on this store -- a note reading "processeur Ryzen
5500U, 32 Go de RAM" is not found by a question that says "matériel".

So rewrite the question as three short search queries, using the
words THE ANSWER would be written with rather than the words of the
question.

Rules:
- Same language as the question.
- No question mark, no politeness, no verb of asking. These are
  search queries, not questions.
- Name things. If the question says "matériel", one rewrite says
  "processeur mémoire disque".
- Under twelve words each, and each different from the other two.
- Rephrase the question. Never answer it, and never invent a detail
  about the person asking -- a name, a brand, a number they did not
  give you.

Question: {query}

EXAMPLE -- form only. Its subject is not yours and its words belong
to it alone. For "Tu peux me parler de la tarte tatin ?":
["recette tarte tatin pommes", "cuisson tarte tatin moule", "tarte tatin caramel beurre"]

Now answer for the real question above, with a JSON array of exactly
three strings and nothing else."""


def _parse(raw: str) -> list[str]:
    """
    The strings out of a model answer.

    Under the grammar this is a plain json.loads, and it is not
    written that way for the same reason spec.parse is not: the
    grammar only exists on llama.cpp. On ollama or OpenRouter the
    same call runs unconstrained and the array arrives inside a fence
    or trailed by a sentence.

    Non-strings are dropped rather than raised on -- a model that
    returns two strings and a number has still produced two usable
    queries.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in the answer")

    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        # ValueError and not TypeError: json.JSONDecodeError is itself
        # a ValueError, and _from_llm treats every unreadable answer
        # the same way. One except clause, one behaviour.
        raise ValueError("the answer is not a JSON array")  # noqa: TRY004
    return [item for item in data if isinstance(item, str)]


def _from_llm(query: str) -> list[str]:
    """
    Ask the model for rephrasings. NEVER raises, and that is the
    contract.

    Everything this returns is a suggestion for where else to look,
    and every one of them is filtered afterwards by the same distance
    cutoff the original query answers to. So the failure modes are
    cheap and they are all the same failure: no extra queries, and a
    recall that behaves exactly as it did before this module existed.
    A provider that is down, a grammar the server refuses, an answer
    that is not an array -- none of them is worth turning into an
    error the user reads, because none of them makes the answer WRONG,
    only unhelped.

    The call is priced honestly in graphs/recall.py: it happens only
    on the path that was about to answer "je n'ai rien en mémoire".
    """
    try:
        raw = call_llm(_PROMPT.format(query=query), grammar=_GRAMMAR)
    except ProviderError as e:
        log.warning("expansion: no rephrasings, the provider failed (%s)", e)
        return []

    try:
        candidates = _parse(raw)
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("expansion: could not read the rephrasings (%s): %r", e, raw[:200])
        return []

    kept = [c for c in candidates if not _echoes_the_example(c, query)]
    log.event(
        "recall.expansion_llm",
        query=query[:120],
        proposed=len(candidates),
        kept=len(kept),
        variants=[c[:80] for c in kept],
    )
    return kept


def _echoes_the_example(candidate: str, query: str) -> bool:
    """
    True when a variant is the worked example coming back instead of
    an answer to the real question.

    Asked of the QUERY first: someone whose store really is about
    baking gets to ask about it, and this check must not be the reason
    their own subject is refused.
    """
    if _EXAMPLE_SUBJECT in _fold(query):
        return False
    if _EXAMPLE_SUBJECT not in _fold(candidate):
        return False
    log.warning(
        "expansion: dropped a rephrasing copied from the prompt example: %r",
        candidate[:80],
    )
    return True
