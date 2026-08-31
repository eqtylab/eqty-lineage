"""The one initialised SDK the whole test session shares.

``eqty_sdk.init()`` is process-global, and a second call does not raise -- it logs ``Config already
initialized`` and returns, leaving the first store in place. So a session fixture per package does not
give each package its own store: ``just test`` runs both packages in one process, whichever is collected
first wins, and the second package's temporary directory is created, chdir'd into, and never written to
while its assets land in the first package's store. That was the arrangement here, and it was invisible
because nothing asserts on where a blob lands.

There is therefore exactly one of these, at the root rather than in either package, and each package's
``conftest.py`` keeps only the ``recording_handler`` that is genuinely its own.
"""

import os

import pytest


@pytest.fixture(scope="session")
def sdk(tmp_path_factory):
    """One initialised SDK context for the whole run, or skip if the SDK is not installed.

    The store is written relative to the working directory, which is pointed at a temporary directory so
    a run deposits no state in the repository.
    """
    eqty_sdk = pytest.importorskip("eqty_sdk", reason="eqty-sdk is not installed")

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
