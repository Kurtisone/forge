"""
The header's model name must not be read once and believed forever.

`checkHealth()` asks /health, which asks llama-server's own /props --
so the name shown is the model actually loaded, not the LLM_MODEL
label. That part was already right. What was wrong is that the UI
called it exactly once, on page load, while `refreshContext()` was
re-called after every turn. Swap the served model and the header goes
on naming the old one for the life of the tab.

Reported after a real swap: "I had to refresh to see the change".

Same approach as test_ui_security.py, and the same caveat: this suite
has no JS runtime, so these are assertions on the source. A regression
tripwire, not a proof -- it catches the realistic failure (someone
removes the call while tidying the turn path) without pulling a JS
engine in for one file.
"""

import re
from pathlib import Path

import forge.api as api_mod

_INDEX = Path(api_mod.__file__).parent / "static" / "index.html"


def _source() -> str:
    return _INDEX.read_text(encoding="utf-8")


def test_health_is_rechecked_after_a_turn():
    """
    The turn path already reloads history and the context gauge. The
    answer just shown came from whatever is loaded now, so the name is
    exactly as stale as the gauge was.
    """
    src = _source()
    turn = src[src.index("async function sendChat") :]
    turn = turn[: turn.index("\n}\n")]
    assert "checkHealth()" in turn, (
        "the send path refreshes the context gauge but not the model "
        "name, which is how the header went stale for a whole session"
    )


def test_health_is_rechecked_when_the_tab_comes_back():
    """
    Swapping a model means leaving this page for a terminal and coming
    back, which is the reported case exactly.
    """
    src = _source()
    assert "visibilitychange" in src
    listener = src[src.index("visibilitychange") :][:200]
    assert "checkHealth()" in listener
    assert "document.hidden" in listener, (
        "visibilitychange fires on hide as well as show; refreshing on "
        "the way out is a request nobody reads"
    )


def test_health_is_not_polled_on_a_timer():
    """
    A tab left open overnight must not keep asking a llama-server it
    has nothing to say to. The two events above are the only ones that
    carry information.
    """
    src = _source()
    for call in re.findall(r"setInterval\s*\([^)]*", src):
        assert "checkHealth" not in call, (
            "checkHealth on a timer polls an idle tab forever; refresh "
            "on the events that can change the answer instead"
        )
