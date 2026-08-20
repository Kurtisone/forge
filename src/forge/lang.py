"""
Which language is this text in? French, English, or don't know.

Exists because "Write in the same language as the question" does not
hold. graphs/recall.py has carried that sentence since it was written
and still answers French questions in English, which is the seventh
time on this codebase that a wording fix lost to something the model
has to infer for itself. Naming the language removes the inference;
checking the answer afterwards removes the trust.

Deliberately NOT a language-detection library. langdetect/langid are
trained on paragraphs and this module is handed one short sentence --
"Le serveur utilise le port 8080." is six words -- where they are
unreliable anyway. A closed function-word table on two languages is
something that can be read, tested and reasoned about at 3am, and
Forge has exactly two languages in play: Kurtisone writes French,
the prompts are English.

The third answer is the important one. UNKNOWN is returned whenever
the evidence is thin or split, and every caller treats it as "say
nothing about language" rather than guessing. Forcing an answer into
the wrong language is a worse failure than the one this module fixes,
because the model obeys.

Adding a language means adding a row to _MARKERS and _NAMES. Nothing
else in here is per-language.
"""

import re
import unicodedata

from forge.logger import log

UNKNOWN = None

# Function words, not vocabulary: they are the words a sentence can't
# avoid, they survive being about an unknown subject, and they don't
# move when the topic is a hostname or a port number.
#
# Words that exist in BOTH languages are excluded on purpose, however
# common they are -- "a", "as", "on", "or", "me", "plus", "son",
# "ton", "as", "car", "no"/"nos". A marker that fires for both sides
# adds noise to both counts and evidence to neither.
_MARKERS = {
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "du",
            "des",
            "une",
            "aux",
            "aux",
            "et",
            "est",
            "sont",
            "était",
            "être",
            "avoir",
            "ai",
            "as",
            "avez",
            "avons",
            "ont",
            "que",
            "qui",
            "quoi",
            "quel",
            "quelle",
            "quels",
            "quelles",
            "pour",
            "dans",
            "sur",
            "avec",
            "sans",
            "sous",
            "chez",
            "vers",
            "pas",
            "ne",
            "rien",
            "jamais",
            "toujours",
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "ce",
            "cette",
            "ces",
            "cet",
            "ça",
            "celui",
            "celle",
            "mon",
            "ma",
            "mes",
            "ton",
            "tes",
            "sa",
            "ses",
            "notre",
            "nos",
            "votre",
            "vos",
            "leur",
            "leurs",
            "au",
            "aux",
            "en",
            "y",
            "donc",
            "mais",
            "car",
            "si",
            "comme",
            "aussi",
            "encore",
            "déjà",
            "ici",
            "là",
            "où",
            "quand",
            "comment",
            "pourquoi",
            "combien",
            "peux",
            "peut",
            "peuvent",
            "fait",
            "faire",
            "fais",
            "dit",
            "dire",
            "tout",
            "toute",
            "toutes",
            "tous",
            "très",
            "bien",
            "alors",
            "même",
            "autre",
            "autres",
            "entre",
            "depuis",
            "pendant",
            "plusieurs",
            "chaque",
            "quelque",
            "quelques",
            "dont",
            "lequel",
            "laquelle",
        ]
    ),
    "en": frozenset(
        [
            "the",
            "of",
            "to",
            "and",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "not",
            "never",
            "always",
            "you",
            "your",
            "yours",
            "i",
            "my",
            "mine",
            "we",
            "our",
            "ours",
            "they",
            "their",
            "them",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "for",
            "with",
            "without",
            "from",
            "into",
            "about",
            "over",
            "under",
            "between",
            "during",
            "since",
            "while",
            "because",
            "so",
            "but",
            "if",
            "then",
            "than",
            "there",
            "here",
            "where",
            "when",
            "how",
            "why",
            "can",
            "could",
            "will",
            "would",
            "should",
            "may",
            "might",
            "must",
            "there's",
            "it's",
            "don't",
            "doesn't",
            "isn't",
            "aren't",
        ]
    ),
}

_NAMES = {"fr": "French", "en": "English"}

# Accents are one-sided evidence: French text is full of them and
# English text has none. Weighted rather than decisive -- a single
# proper noun ("Forgejo déployé") should tip a tie, not overrule six
# English function words.
_ACCENT_WEIGHT = 2

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def _has_accents(text: str) -> bool:
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", text))


def score(text: str) -> dict[str, int]:
    """Marker hits per language. Exposed so a caller can log why."""
    words = [w.lower() for w in _WORD.findall(text or "")]
    scores = {
        code: sum(1 for w in words if w in markers)
        for code, markers in _MARKERS.items()
    }
    if _has_accents(text or ""):
        scores["fr"] += _ACCENT_WEIGHT
    return scores


def detect(text: str) -> str | None:
    """
    "fr", "en", or None when the evidence doesn't decide.

    None on a tie, and None on no evidence at all: a hostname, a port
    number, a bare "ok". Callers must treat it as "don't mention
    language", never as a default.
    """
    scores = score(text)
    best = max(scores, key=lambda code: scores[code])
    runner_up = max((s for c, s in scores.items() if c != best), default=0)
    if scores[best] == 0 or scores[best] == runner_up:
        return UNKNOWN
    return best


def name(code: str | None) -> str | None:
    """The English name of a code, for putting inside a prompt."""
    return _NAMES.get(code) if code else None


def mismatch(expected_from: str, answer: str) -> str | None:
    """
    The name of the language `answer` SHOULD have been in, when it
    demonstrably isn't.

    Returns None unless both texts are confidently detected AND they
    disagree -- three separate ways of saying "no", because acting on
    this costs a second LLM call and, worse, tells the model its
    correct answer was wrong.
    """
    want = detect(expected_from)
    got = detect(answer)
    if want is None or got is None or want == got:
        return None
    return _NAMES[want]


# ---------------------------------------------------------------------
# The two halves of "answer in the question's language", shared.
#
# recall shipped these as module-local strings in v3.12's dettes batch.
# review, research and sysadmin need the same thing -- their synthesis
# prompts are English prose bodies pulling a French answer toward
# English exactly the way recall's was -- and four copies of a wording
# that has to stay identical is how a fix drifts. The strings live
# here; each graph keeps its own prompt building, because they differ.
# ---------------------------------------------------------------------

# Last position, and only ever a language detect() was sure about. The
# instruction this replaced ("Write in the same language as the
# question") sat mid-paragraph and asked the model to infer the target
# for itself; it answered French questions in English anyway.
_LINE = "\nWrite your answer in {language}. Every word of it.\n"

# Second pass only. Says what was wrong, because "do it again" on its
# own is what the model just did.
_RETRY_LINE = (
    "\nYour previous answer was in the wrong language. The question is "
    "in {language}. Write the answer again, in {language}, every word "
    "of it -- same content, same length.\n"
)


def line_for(question: str) -> str:
    """
    The instruction to append LAST to a synthesis prompt, or "" when
    the question's language is uncertain.

    Empty rather than a default, and that is the whole design: forcing
    an answer into the wrong language is worse than the bug, because
    the model complies.
    """
    language = name(detect(question))
    return _LINE.format(language=language) if language else ""


def enforce(question: str, answer: str, retry, *, enabled: bool = True) -> str:
    """
    The deterministic half. `retry` is called with a replacement
    language line and must return a cleaned answer, so each caller
    keeps its own prompt shape and its own response cleaning.

    Naming the language in the prompt is still a wording fix, and
    wording fixes have lost seven times on this codebase. This is the
    part that does not depend on the model having complied. It costs
    one extra call, and only on the runs that were already wrong.

    At most one retry, never on an "[error] ..." answer, never when
    either language is uncertain, and the FIRST answer is kept unless
    the second is both usable and actually in the right language: an
    answer whose language is wrong still has the right content, which
    is more than an error message has.
    """
    if not enabled or answer.startswith("[error]"):
        return answer

    wanted = mismatch(question, answer)
    if not wanted:
        return answer

    log.warning("answered in the wrong language (question is %s) -- retrying", wanted)
    second = retry(_RETRY_LINE.format(language=wanted))
    if not second.startswith("[error]") and not mismatch(question, second):
        return second

    log.warning("retry did not fix the language, keeping the first")
    return answer
