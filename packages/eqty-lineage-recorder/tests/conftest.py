"""Fixtures for the recorder tests.

The session-scoped ``sdk`` fixture lives in the workspace root ``conftest.py``, not here. ``init()`` is
process-global and a second call is ignored rather than refused, so a session fixture per package does
not give each package its own store -- ``just test`` runs every package in one process, whichever is
collected first wins, and the rest quietly record into its store. One fixture, at the root.
"""

import pytest


@pytest.fixture
def recorder(sdk):
    """A recorder bound to the session context.

    Recording happens inside ``graph_context`` because the SDK resolves context differently for assets
    and statements: asset constructors fall back to the active context, statement constructors do not.
    """
    from eqty_lineage.recorder import LineageRecorder
    from eqty_sdk.context import graph_context

    with graph_context(sdk):
        yield LineageRecorder(framework="test")
