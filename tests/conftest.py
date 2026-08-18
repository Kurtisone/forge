"""
Shared pytest fixtures.

Every test gets an isolated memory file and jobs file automatically,
so running the test suite never reads or writes the real
data/memory.json or data/jobs.json in the repo (and tests never bleed
state into each other through them).
"""

import pytest

from forge import jobs, memory


@pytest.fixture(autouse=True)
def isolated_memory_file(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(jobs, "JOBS_FILE", str(tmp_path / "jobs.json"))
    yield
