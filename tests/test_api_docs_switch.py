"""
Tests for the interactive docs switch (security audit, M-3).

/docs, /redoc and /openapi.json are mounted by FastAPI itself rather
than by this app's routes, so there is no way to hang
Depends(require_token) on them -- the only two states available are
"published to anyone who can reach the port" and "not mounted".
API_DOCS_ENABLED picks between them, defaulting to not mounted.

The enabled case needs a separate interpreter: docs_url is read once,
when the FastAPI object is constructed at import time, so
monkeypatching the flag afterwards changes nothing. Reloading
forge.api in-process would hand the rest of the suite a different
module object than the one it holds a reference to, so a subprocess
is the honest way to exercise that branch.
"""

import json
import subprocess
import sys
import textwrap

from fastapi.testclient import TestClient

import forge.api as api_mod

_DOC_ROUTES = ["/docs", "/redoc", "/openapi.json"]


def _statuses_in_subprocess(env_value: str | None) -> dict[str, int]:
    script = textwrap.dedent(
        f"""
        import json
        from fastapi.testclient import TestClient
        import forge.api as api_mod

        client = TestClient(api_mod.app)
        routes = {_DOC_ROUTES!r}
        print(json.dumps({{route: client.get(route).status_code for route in routes}}))
        """
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": ":".join(sys.path),
        # The app refuses to start unauthenticated; this subprocess only
        # builds the app object, but keep it configured either way.
        "API_TOKEN": "s3cret",
    }
    if env_value is not None:
        env["API_DOCS_ENABLED"] = env_value

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_docs_are_not_mounted_by_default():
    """The default has to be checked against a fresh interpreter with
    no API_DOCS_ENABLED in the environment -- asserting it against the
    already-imported app would only prove what this test run happens
    to be configured as."""
    statuses = _statuses_in_subprocess(None)
    assert all(status == 404 for status in statuses.values()), statuses


def test_docs_are_mounted_when_explicitly_enabled():
    """The off switch is only defensible if the on switch works;
    otherwise the next person needing the schema turns off something
    else instead."""
    statuses = _statuses_in_subprocess("true")
    assert all(status == 200 for status in statuses.values()), statuses


def test_the_app_under_test_matches_its_own_configuration():
    """Ties the running app object to the flag it was built from, so
    this file can't pass while the real app quietly publishes the
    schema."""
    client = TestClient(api_mod.app)
    expected = 200 if api_mod.API_DOCS_ENABLED else 404
    for route in _DOC_ROUTES:
        assert client.get(route).status_code == expected, route
