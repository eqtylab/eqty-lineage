"""Shared fixtures.

Two rules shape this file.

**Fixtures are synthetic.** Real Claude Code transcripts under ``~/.claude/projects/`` carry the
contents of whatever repository the session touched, along with absolute paths and prompts. None of that
can be committed, so the transcript fixture is hand-built to exercise the parser's edge cases and the
Codex fixture is a real capture with paths and home directory rewritten.

**``eqty_sdk.init()`` is process-global and raises on a second call.** Tests that need the SDK share one
session-scoped context rather than initialising per test, and tests that do not need it must not import
it -- the parser, the semirings and the engine are all exercised without a signer.
"""

import pytest


@pytest.fixture(scope="session")
def sdk(tmp_path_factory):
    """One initialised SDK context for the whole run, or skip if the SDK is not installed.

    ``init()`` writes a ``.eqty_sdk`` store relative to the working directory, so it is pointed at a
    temporary directory to keep test runs from depositing state in the repository.
    """
    eqty_sdk = pytest.importorskip("eqty_sdk", reason="eqty-sdk is not installed")

    import os

    store = tmp_path_factory.mktemp("eqty-store")
    previous = os.getcwd()
    os.chdir(store)
    try:
        ctx = eqty_sdk.Context.new("eqty-lineage tests")
        eqty_sdk.init(default_context=ctx).set_store_all_blobs(True)
        eqty_sdk.set_active_signer(eqty_sdk.Signer.new(name="eqty-lineage-tests", _load_if_exists=True))
        yield ctx
    finally:
        os.chdir(previous)


@pytest.fixture
def recorder(sdk):
    """A recorder bound to the session context.

    Recording happens inside ``graph_context`` because the SDK resolves context differently for assets
    and statements: asset constructors fall back to the active context, statement constructors do not.
    """
    from eqty_lineage.core import LineageRecorder
    from eqty_sdk.context import graph_context

    with graph_context(sdk):
        yield LineageRecorder(framework="test")
