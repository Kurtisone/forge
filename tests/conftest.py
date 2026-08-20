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


@pytest.fixture(autouse=True)
def permissive_policy(monkeypatch):
    """
    Pin the Policy Engine open for every test.

    Dispatch consults the policy, so without this the suite's result
    would depend on whoever ran it: a developer with
    POLICY_ALLOW_NETWORK=false in their .env.local would watch
    unrelated orchestrator tests fail for a reason the assertion never
    mentions. Same class of bug as a test inheriting ENABLED_TOOLS from
    the environment -- pinned here once rather than in each test.

    Tests that need a restriction switch these off themselves; a
    monkeypatch applied inside the test overrides this one and is
    undone first.
    """
    from forge.kernel import policy

    for flag in (
        "POLICY_ALLOW_NETWORK",
        "POLICY_ALLOW_WORKSPACE_WRITES",
        "POLICY_ALLOW_SUBPROCESS",
    ):
        monkeypatch.setattr(policy, flag, True)
    yield
