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


@pytest.fixture(scope="module")
def ip():
    return _load("instruct_prefix")


class TestReadRow:
    """
    The A/B's subject is the entry the operator named, not the row that
    happened to win. On 2026-08-23 those were different rows for "Tu
    peux me lister mon matériel ?" and the difference was the result.
    """

    def test_the_named_entry_is_what_gets_scored(self, rd):
        assert rd.read_row(ROWS, "308") == (0.9741, 2, 0.9083)

    def test_the_named_entry_winning_makes_the_two_agree(self, rd):
        assert rd.read_row(ROWS, "176") == (0.9083, 1, 0.9083)

    def test_nothing_named_falls_back_to_the_closest_row(self, rd):
        assert rd.read_row(ROWS, None) == (0.9083, None, 0.9083)

    def test_a_named_entry_that_never_came_back_reports_no_rank(self, rd):
        # The distance is the closest row's, so the line still prints
        # something -- and rank None is what tells the caller not to
        # put that number in the verdict.
        assert rd.read_row(ROWS, "999") == (0.9083, None, 0.9083)

    def test_no_results_at_all(self, rd):
        assert rd.read_row([], "308") == (None, None, None)


class TestScoreable:
    def test_the_named_entry_came_back_second_and_still_counts(self, ip):
        assert ip._scoreable("308", (0.9741, 2, 0.9083))

    def test_the_named_entry_never_came_back_and_does_not(self, ip):
        # The fault this harness was carrying: that 0.9083 is the
        # distance to an entry about tools, and it went into the gap
        # as though it were a hardware hit.
        assert not ip._scoreable("308", (0.9083, None, 0.9083))

    def test_nothing_named_means_the_closest_row_is_all_there_is(self, ip):
        assert ip._scoreable(None, (0.9083, None, 0.9083))

    def test_no_distance_is_never_scoreable(self, ip):
        assert not ip._scoreable(None, (None, None, None))


class TestCell:
    def test_a_missing_row_does_not_crash_the_format(self, ip):
        assert ip._cell((None, None, None)).strip() == "--"

    def test_the_closest_row_stays_visible_beside_the_scored_one(self, ip):
        assert ip._cell((0.9741, 2, 0.9083)).startswith("0.9741 rank=2")
        assert "[0.9083]" in ip._cell((0.9741, 2, 0.9083))
