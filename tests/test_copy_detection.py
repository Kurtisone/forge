"""
Tests for the "the answer is just the input" guard.

Run #b669174a, live on the Deck: `review` was pointed at a 20 000
character file, saw the first 8 000, and answered with the file's own
opening docstring copied verbatim -- 97 words of input handed back as
analysis.

Nothing caught it, and every guard was right by its own terms.
try_unwrap_router_json checks that unwrapped content is SUBSTANTIVE
(>= 8 words, >= 40 chars) so that a degenerate one-word echo is not
mistaken for an answer; a verbatim copy of the input is maximally
substantive. _PROMPT_LEAK_MARKERS looks for the prompt's phrases, and
this copied the file. sysadmin's _EXAMPLE_LEAK_FRAGMENTS looks for the
GOOD ANSWER example -- the same failure against a third source text.

The thing all three point at is checkable without a model: "did this
answer come out of that material" is a string question.
"""

from forge.text_cleaning import looks_like_a_copy

# A real docstring-shaped block, the shape that was actually copied.
SOURCE = """
The router prompt lives here and ONLY here.

If you ever need to tweak how the router is instructed, this is the
single file to touch -- nothing else in the codebase builds or
concatenates prompt text.

The prompt is generated dynamically from the currently enabled tools
(forge.tools.registry.available_tools()), not a fixed chat/code pair.
A tool the operator hasn't opted into via ENABLED_TOOLS never appears
in the prompt, so the router can't be steered toward offering it.
"""


def test_a_verbatim_copy_is_caught():
    assert looks_like_a_copy(SOURCE.strip(), SOURCE)


def test_a_reflowed_copy_is_still_caught():
    """
    A copy is rarely clean -- the model reflows lines and drops a
    paragraph. An exact-match check would miss every real case.
    """
    mangled = " ".join(SOURCE.split())
    mangled = mangled.replace("A tool the operator hasn't opted into", "")

    assert looks_like_a_copy(mangled, SOURCE)


def test_a_copy_with_a_sentence_bolted_on_is_still_caught():
    assert looks_like_a_copy(SOURCE.strip() + "\n\nOverall this looks fine.", SOURCE)


def test_a_real_review_that_quotes_the_file_is_not_caught():
    """
    The false positive that matters. Quoting the material is correct
    behaviour -- a review cites the offending line, a diagnosis
    reproduces the log entry it explains -- and a guard that punishes
    it would be worse than the bug.
    """
    review = (
        "Ce module concentre toute la construction du prompt, ce qui est "
        "une bonne décision : un seul endroit à modifier. La docstring "
        "annonce que « The prompt is generated dynamically from the "
        "currently enabled tools », et c'est bien ce que fait le code. "
        "En revanche rien ne vérifie qu'un outil listé dans ENABLED_TOOLS "
        "expose réellement un handler, donc une faute de frappe dans la "
        "configuration produit un prompt silencieusement amputé plutôt "
        "qu'une erreur au démarrage. Je suggère une validation explicite "
        "au chargement, et un test qui compare les deux listes."
    )

    assert not looks_like_a_copy(review, SOURCE)


def test_an_unrelated_answer_is_not_caught():
    answer = (
        "Le fichier est globalement clair mais la fonction principale "
        "fait trop de choses à la fois : elle valide, formate et écrit. "
        "Je la découperais en trois, ce qui rendrait chaque morceau "
        "testable séparément et supprimerait le besoin du drapeau booléen."
    )

    assert not looks_like_a_copy(answer, SOURCE)


def test_a_short_answer_is_exempt():
    """
    Below one window there is nothing to measure, and a legitimate
    one-line answer can easily be a phrase that also occurs in the
    source.
    """
    assert not looks_like_a_copy("The router prompt lives here.", SOURCE)


def test_no_source_means_no_opinion():
    """
    Callers that have no material to compare against (a graph entered
    directly, a test) must not be silently failed.
    """
    assert not looks_like_a_copy(SOURCE.strip(), "")


def test_review_rejects_an_answer_that_is_the_file():
    from forge.graphs import review

    cleaned = review._clean_review_response(SOURCE.strip(), SOURCE)

    assert cleaned.startswith("[error]")
    assert "recopié" in cleaned


def test_review_still_accepts_a_real_review_of_that_file():
    from forge.graphs import review

    answer = (
        "La responsabilité est bien isolée dans un seul module, ce qui "
        "évite la dispersion du texte de prompt. Deux réserves : rien ne "
        "valide ENABLED_TOOLS au démarrage, et les descriptions sont "
        "payées à chaque tour même quand l'outil n'est jamais choisi."
    )

    assert review._clean_review_response(answer, SOURCE) == answer


def test_sysadmin_rejects_an_answer_that_is_the_log_block():
    from forge.graphs import sysadmin

    logs = "\n".join(
        f"aug 21 10:0{i} deck kernel: overlayfs: xino=off" for i in range(9)
    )
    cleaned = sysadmin._clean_diagnosis_response(logs, logs)

    assert cleaned.startswith("[error]")
