"""
The measuring instrument, measured.

WHY THIS FILE EXISTS

Five faults reached the Deck during the 2026-08-23 session. Every one
of them was in code no test ran:

  - rag._embed_one sent the instruction where the URL goes. 1145 tests
    passed; all of them stub rag._embed, so the HTTP call had no
    coverage at all.
  - bench/recall_distance.py's --expect looked for the entry id in the
    field holding fixture text, so rank was None for every question in
    every run since it shipped.
  - bench/instruct_prefix.py compared one instruction against two,
    because rag.search had started applying one itself.
  - the misplaced-entry warning tested `rank != 1` while excluding
    None, so the one case it was written for -- the expected entry not
    coming back at all -- passed through it silently.

Four of the five were fixed the same day. The fifth was that
bench/instruct_prefix.py carried its own copy of the same reading and
none of those fixes reached it, so the harness that justified turning
the query instruction on went on scoring a failed retrieval as a
mediocre hit. The copies are one module now (bench/_harness.py), which
is why these tests load that instead of a specific harness.

The pattern is not carelessness in five places, it is that bench/ had
no tests and everything in it was verified by reading. A harness whose
job is to say whether a change worked is exactly the code where a
silent fault is most expensive: it does not crash, it prints a
confident number, and the number is believed.

So the decisions in these scripts are pure functions now, and this
file exercises them. The scripts still need a real store and a real
embedding server to run; these two decisions do not.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "bench"


def _load(name: str):
    # bench/ goes on the path because the harnesses import _harness
    # from their own directory -- which is how they run in the
    # container, where the script's directory is sys.path[0].
    if str(_BENCH) not in sys.path:
        sys.path.insert(0, str(_BENCH))
    spec = importlib.util.spec_from_file_location(name, _BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rd():
    """The shared decisions, which is where these two now live."""
    return _load("_harness")


ROWS = [
    {"id": 176, "content": "user: Tu as accès à quels outils ?", "distance": 0.9083},
    {"id": 308, "content": "NiPoGi AM06PRO, Ryzen 5500U", "distance": 0.9741},
]


class TestFindRank:
    def test_finds_a_planted_fixture_by_its_text(self, rd):
        assert rd.find_rank(ROWS, "NiPoGi AM06PRO, Ryzen 5500U", None) == 2

    def test_finds_an_expected_entry_by_its_id(self, rd):
        assert rd.find_rank(ROWS, None, "176") == 1
        assert rd.find_rank(ROWS, None, "308") == 2

    def test_an_id_given_as_an_int_still_matches(self, rd):
        assert rd.find_rank(ROWS, None, 308) == 2

    def test_absent_from_the_results_is_none(self, rd):
        assert rd.find_rank(ROWS, None, "999") is None

    def test_no_expectation_is_none(self, rd):
        assert rd.find_rank(ROWS, None, None) is None

    def test_a_row_with_no_content_does_not_crash(self, rd):
        assert rd.find_rank([{"id": 1, "content": None}], "text", None) is None


class TestMisplaced:
    def test_first_place_is_fine(self, rd):
        assert rd.misplaced([("q", "308", 1)]) == []

    def test_second_place_is_reported(self, rd):
        assert rd.misplaced([("q", "308", 2)]) == ["q"]

    def test_absent_entirely_is_reported(self, rd):
        # The case the first version let through: `rank != 1` written
        # as `r is not None and r != 1`. A question whose answer was
        # nowhere in the results passed silently and its distance went
        # into the gap as though it were a hit.
        assert rd.misplaced([("q", "308", None)]) == ["q"]

    def test_no_expectation_is_not_a_failure(self, rd):
        assert rd.misplaced([("q", None, None)]) == []

    def test_the_real_run(self, rd):
        # 2026-08-23, the three hit questions as measured.
        rows = [
            ("Quel processeur a mon NiPoGi ?", "308", 1),
            ("Tu peux me lister mon matériel ?", "308", None),
            ("Quels outils as-tu accès ?", "176", 1),
        ]
        assert rd.misplaced(rows) == ["Tu peux me lister mon matériel ?"]


class TestPlaceholders:
    def test_a_forgotten_slot_is_caught(self, rd):
        assert rd.placeholders(["<la 3e question hit du 22/08>"])

    def test_a_real_question_is_not(self, rd):
        assert rd.placeholders(["Quel processeur a mon NiPoGi ?"]) == []

    def test_an_ellipsis_is_caught(self, rd):
        assert rd.placeholders(["Quel est le ..."])
