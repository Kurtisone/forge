"""Tests for forge.context_info (shared date injection for prompts)."""

from datetime import date

import forge.context_info as ci


def test_today_line_contains_current_iso_date():
    line = ci.today_line()
    assert date.today().isoformat() in line  # noqa: DTZ011


def test_today_line_format():
    line = ci.today_line()
    assert line.startswith("Today's date is ")
    assert line.endswith(".")
