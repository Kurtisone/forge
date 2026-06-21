"""
Shared pytest fixtures.

Every test gets an isolated memory file automatically, so running
the test suite never reads or writes the real data/memory.json in
the repo (and tests never bleed state into each other through it).
"""

import pytest

import forge.memory as memory


@pytest.fixture(autouse=True)
def isolated_memory_file(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    yield
