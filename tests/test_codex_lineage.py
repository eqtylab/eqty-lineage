import json
import base64
from pathlib import Path

from eqty_lineage.codex import build_demo


def test_codex_demo_exports_signed_allow_and_deny_graph(tmp_path: Path):
    manifest = json.loads(Path(build_demo(tmp_path / "codex.json")).read_text())
    metadata = []
    for statement in manifest["statements"].values():
        if statement.get("@type") == "MetadataRegistration":
            metadata.append(json.loads(base64.b64decode(manifest["blobs"][statement["metadata"].replace("urn:cid:", "")])) | statement)
    assert metadata
    assert all(item["registeredBy"].startswith("did:key:") for item in metadata)
    assert sum(item.get("name") == "Codex user prompt" for item in metadata) == 1
    assert sum(item.get("decision") == "allow" for item in metadata) == 1
    assert sum(item.get("decision") == "deny" for item in metadata) == 1
    assert sum(item.get("computation_type") == "tool" for item in metadata) == 2
