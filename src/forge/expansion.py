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

The half that can bridge vocabulary needs a model, and it is the
next layer -- built on top of this one rather than instead of it.
"""

import re
import unicodedata

from forge.logger import log

#: The modes RECALL_EXPANSION accepts.
MODES = ("off", "terms")

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

    No model, no network, no configuration. Either can come back
    identical to the query or to each other, in which case `keep`
    drops it and the caller pays nothing.

    The second is derived from the FIRST and not from the raw query:
    the frame is a phrase, not a bag of stopwords, and "s'il te
    plaît" survives a stopword filter as `plaît` -- a word no stored
    fact has ever contained.
    """
    unframed = _without_frame(query)
    return keep([unframed, _content_words(unframed)], query)


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
    log.warning(
        "unknown RECALL_EXPANSION=%r, expansion is off (expected one of: %s)",
        mode,
        ", ".join(MODES),
    )
    return []
