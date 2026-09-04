"""The extension point a framework package uses to teach the handler about its own state.

Everything a node's state holds is otherwise serialized into that node's state Dataset, every time -- so a
filesystem carried in state is embedded once per node, and no file is ever an entity in its own right. An
extractor claims part of a state, registers whatever assets represent it, and returns what should stand in
its place in the payload.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from eqty_sdk import CID, Dataset, get_cid_for_path

from eqty_lineage.langchain._serialize import UNCLAIMED

if TYPE_CHECKING:
    from eqty_lineage.langchain import EqtyCallbackHandler

logger = logging.getLogger("eqty.langgraph")


class AssetSink:
    """Where an extractor records the assets it produced, and why each one is there.

    *Carried* means the entity already existed and this computation merely handled it, so it is an input.
    *Created* means this computation produced it, so it is an output. Getting the distinction wrong is how
    a lineage graph grows a cycle.
    """

    def __init__(self, metadata: Dict[str, Any]) -> None:
        #: sanitized verbose metadata, ready to unpack into an SDK asset constructor
        self.metadata = metadata
        self.carried: List[CID] = []
        self.created: List[CID] = []

    def carry(self, cid: CID) -> None:
        if cid not in self.carried and cid not in self.created:
            self.carried.append(cid)

    def create(self, cid: CID) -> None:
        if cid not in self.created and cid not in self.carried:
            self.created.append(cid)


class StateExtractor:
    """Lifts part of a graph state into assets of its own, and out of the bulk state blob.

    Claim a value, register the assets that represent it, return what should stand in its place. Subclass
    and register with ``EqtyCallbackHandler.add_extractor`` to teach the handler about a framework's own
    state without this package importing that framework.
    """

    def extract(self, key_path: Tuple[str, ...], value: Any, sink: AssetSink) -> Any:
        """Claim ``value`` and return its replacement, or ``UNCLAIMED`` to decline.

        ``key_path`` is the sequence of dict keys that led here, so an extractor can claim a particular
        state key (``("files",)``) rather than guessing from the value's shape.
        """
        raise NotImplementedError


class PathExtractor(StateExtractor):
    """Registers an existing ``pathlib.Path`` in state as a Dataset of its own.

    Keyed on ``(path, content CID)``, not the path alone: reusing the first sighting's asset would attest
    content a later computation never saw. The version a rewrite replaced is carried as an input, which
    makes successive edits a chain.
    """

    def __init__(self, handler: "EqtyCallbackHandler") -> None:
        self._handler = handler

    def extract(self, key_path: Tuple[str, ...], value: Any, sink: AssetSink) -> Any:
        if not isinstance(value, Path):
            return UNCLAIMED
        if not value.exists():
            return str(value)

        key = str(value.resolve())
        try:
            content_cid = str(get_cid_for_path(value))
        except Exception:  # noqa: BLE001 - an unreadable path must not take down the run being observed
            logger.debug("could not compute a content CID for '%s'; skipping", value)
            return str(value)

        known = self._handler._path_versions.get((key, content_cid))
        if known is not None:
            sink.carry(known)
            return str(value)

        asset = self._handler._asset_factory(Dataset).from_path(
            value,
            name=value.name,
            description=f"Filesystem asset referenced by LangGraph state: '{value}'.",
            **sink.metadata,
        )
        self._handler._path_versions[(key, content_cid)] = asset.cid

        previous = self._handler._path_latest.get(key)
        if previous is not None:
            sink.carry(previous)
        self._handler._path_latest[key] = asset.cid

        sink.create(asset.cid)
        return str(value)
