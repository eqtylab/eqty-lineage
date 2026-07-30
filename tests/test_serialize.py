"""Payload coercion, and the byte-stability the whole content-addressing scheme rests on.

`as_bytes` decides what gets hashed into a file version's CID. If it is not stable across capture
paths, the same file seen live and seen in a transcript hashes differently, the equivalence check
fails, and -- worse -- the graph grows a spurious version that no reader can tell from a real edit. It
had no targeted tests despite being exercised on every asset in the package.

`scalar_metadata` encodes hard-won rules about what the SDK constructors and the graph explorer
tolerate. The collision handling in particular exists because `Dataset.from_object(..., name=...)`
raises "got multiple values for keyword argument" when caller metadata shadows a reserved kwarg.
"""

import json
from pathlib import Path

from eqty_lineage.core import as_bytes, scalar_metadata, to_jsonable
from eqty_lineage.core.serialize import COLLISION_PREFIX, RESERVED_SDK_KWARGS


class TestAsBytesIsStable:
    """The property CIDs depend on: same logical content, same bytes, every time."""

    def test_a_dict_hashes_the_same_regardless_of_key_order(self):
        # Two capture paths can build the same mapping in different orders. If this were order
        # dependent, they would disagree on the CID and the graph would grow a phantom version.
        assert as_bytes({"a": 1, "b": 2}) == as_bytes({"b": 2, "a": 1})

    def test_nested_dicts_are_also_order_independent(self):
        left = {"outer": {"x": 1, "y": 2}, "z": 3}
        right = {"z": 3, "outer": {"y": 2, "x": 1}}
        assert as_bytes(left) == as_bytes(right)

    def test_it_is_deterministic_across_calls(self):
        value = {"path": "/repo/a.py", "lines": [1, 2, 3]}
        assert as_bytes(value) == as_bytes(value)

    def test_bytes_pass_through_untouched(self):
        assert as_bytes(b"\x00raw\xff") == b"\x00raw\xff"

    def test_a_string_is_utf8_encoded(self):
        assert as_bytes("héllo") == "héllo".encode("utf-8")

    def test_none_is_empty_rather_than_the_string_none(self):
        # "None" would be four bytes of content the file never had.
        assert as_bytes(None) == b""

    def test_the_encoding_is_compact(self):
        # Separators without spaces: whitespace differences would change the CID for identical data.
        assert as_bytes({"a": 1}) == b'{"a":1}'

    def test_different_content_gives_different_bytes(self):
        assert as_bytes({"a": 1}) != as_bytes({"a": 2})


class TestToJsonable:
    def test_scalars_survive_unchanged(self):
        for value in (None, "s", 1, 1.5, True):
            assert to_jsonable(value) == value

    def test_bytes_degrade_to_text_rather_than_exploding(self):
        # Binary payloads must not take down the session being observed.
        assert to_jsonable(b"\xff\xfeok") == "��ok"

    def test_paths_become_strings(self):
        assert to_jsonable(Path("/repo/a.py")) == "/repo/a.py"

    def test_nested_structures_are_converted_recursively(self):
        assert to_jsonable({"k": [Path("/a"), b"b"]}) == {"k": ["/a", "b"]}

    def test_tuples_and_sets_become_lists(self):
        assert to_jsonable((1, 2)) == [1, 2]
        assert to_jsonable({1}) == [1]

    def test_dict_keys_are_stringified(self):
        assert to_jsonable({1: "a"}) == {"1": "a"}

    def test_an_object_with_a_dict_is_unpacked(self):
        class Thing:
            def __init__(self):
                self.x = 1

        assert to_jsonable(Thing()) == {"x": 1}

    def test_an_unconvertible_object_falls_back_to_str(self):
        class Opaque:
            __slots__ = ()

            def __str__(self):
                return "opaque"

        assert to_jsonable(Opaque()) == "opaque"

    def test_a_model_dump_is_preferred_when_present(self):
        class Model:
            def model_dump(self):
                return {"from": "model_dump"}

        assert to_jsonable(Model()) == {"from": "model_dump"}

    def test_a_failing_model_dump_falls_through_rather_than_raising(self):
        class Broken:
            def model_dump(self):
                raise RuntimeError("nope")

        # Falls through to __dict__ handling; the point is that it does not propagate.
        assert to_jsonable(Broken()) is not None


class TestJsonableHook:
    """The adapter escape hatch: core must not need to know about a LangChain message or a Claude
    Code content block."""

    def test_the_hook_is_tried_first(self):
        assert to_jsonable({"a": 1}, lambda o: "handled") == "handled"

    def test_not_implemented_defers_to_the_default(self):
        assert to_jsonable({"a": 1}, lambda o: NotImplemented) == {"a": 1}

    def test_the_hook_reaches_nested_values(self):
        def hook(obj):
            return "X" if obj == 1 else NotImplemented

        assert to_jsonable({"k": [1, 2]}, hook) == {"k": ["X", 2]}


class TestScalarMetadata:
    def test_scalars_are_left_alone(self):
        assert scalar_metadata({"n": 1, "s": "x", "b": True}) == {"n": 1, "s": "x", "b": True}

    def test_non_scalars_are_json_encoded(self):
        # The graph explorer renders each metadata value as a string; an un-encoded dict shows up as
        # "[object Object]".
        assert scalar_metadata({"d": {"a": 1}}) == {"d": json.dumps({"a": 1})}

    def test_none_is_kept_and_encoded(self):
        # A present-but-empty key has to stay distinguishable from an absent one.
        assert scalar_metadata({"k": None}) == {"k": "null"}

    def test_a_reserved_kwarg_is_prefixed_until_it_is_not(self):
        # Without this, Dataset.from_object(..., name=...) raises "got multiple values for keyword
        # argument" as soon as caller metadata carries a `name`.
        for reserved in RESERVED_SDK_KWARGS:
            out = scalar_metadata({reserved: "v"})
            assert reserved not in out
            assert f"{COLLISION_PREFIX}{reserved}" in out

    def test_keys_are_deduplicated_against_each_other(self):
        # Two keys that collide after prefixing must not silently overwrite one another.
        out = scalar_metadata({"name": "a", "x-name": "b"})
        assert len(out) == 2
        assert set(out.values()) == {"a", "b"}

    def test_a_custom_reserved_set_is_honoured(self):
        out = scalar_metadata({"mine": 1}, reserved={"mine"})
        assert out == {f"{COLLISION_PREFIX}mine": 1}

    def test_the_hook_applies_here_too(self):
        assert scalar_metadata({"k": object()}, lambda o: "hooked") == {"k": "hooked"}

    def test_an_empty_mapping_stays_empty(self):
        assert scalar_metadata({}) == {}
