"""Tests for forge.text_cleaning (shared by graphs/review.py and
graphs/research.py -- see its module docstring for why this was
centralized after the same bug appeared independently in both)."""

import forge.text_cleaning as tc


def test_strip_think_blocks():
    assert tc.strip_think_blocks("<think>reasoning</think>Answer.") == "Answer."


def test_strip_think_blocks_no_think_block():
    assert tc.strip_think_blocks("Just an answer.") == "Just an answer."


def test_try_unwrap_router_json_substantive_content():
    wrapped = '{"tool":"chat","content":"This is a real, substantive answer with many words in it."}'
    result = tc.try_unwrap_router_json(wrapped, source="test")
    assert result == "This is a real, substantive answer with many words in it."


def test_try_unwrap_router_json_degenerate_content():
    wrapped = '{"tool":"chat","content":"hello.go"}'
    result = tc.try_unwrap_router_json(wrapped, source="test")
    assert result is None


def test_try_unwrap_router_json_not_json():
    assert tc.try_unwrap_router_json("just plain text", source="test") is None


def test_try_unwrap_router_json_json_but_no_content_field():
    assert tc.try_unwrap_router_json('{"tool":"chat"}', source="test") is None


def test_try_unwrap_router_json_content_not_a_string():
    assert (
        tc.try_unwrap_router_json('{"tool":"chat","content":123}', source="test")
        is None
    )
