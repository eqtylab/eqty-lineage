"""Independent local attestation must be compared against, never replaced by, an external service's claims."""


def _handler_with_deferred_run(monkeypatch, input_cid, output_cid):
    from eqty_lineage.langchain import EqtyCallbackHandler

    handler = EqtyCallbackHandler(defer_chat_model_computations=True)
    handler._completed_chat_models.append({"name": "vNIM", "inputs": [input_cid], "output": output_cid})

    calls = []
    monkeypatch.setattr(
        handler, "_finalize", lambda name, kind, inputs, outputs: calls.append((name, kind, inputs, outputs))
    )
    return handler, calls


def test_matching_cids_link_straight_through(sdk, monkeypatch):
    from eqty_sdk import get_cid_for_bytes

    input_cid = get_cid_for_bytes(b"input")
    output_cid = get_cid_for_bytes(b"output")
    handler, calls = _handler_with_deferred_run(monkeypatch, input_cid, output_cid)

    request_cid = get_cid_for_bytes(b"the exact request bytes")
    response_cid = get_cid_for_bytes(b"the exact response bytes")

    handler.register_external_chat_transport(
        local_request_cid=request_cid,
        vnim_request_cid=request_cid,
        vnim_response_cid=response_cid,
        local_response_cid=response_cid,
        name="vNIM",
    )

    kinds = [kind for _, kind, _, _ in calls]
    assert kinds == ["external_request", "external_response"]
    assert calls[0][2:] == ([input_cid], [request_cid])
    assert calls[1][2:] == ([response_cid], [output_cid])


def test_request_mismatch_is_flagged_and_does_not_raise(sdk, monkeypatch, caplog):
    from eqty_sdk import get_cid_for_bytes

    input_cid = get_cid_for_bytes(b"input")
    output_cid = get_cid_for_bytes(b"output")
    handler, calls = _handler_with_deferred_run(monkeypatch, input_cid, output_cid)

    local_request_cid = get_cid_for_bytes(b"what this app actually sent")
    vnim_request_cid = get_cid_for_bytes(b"what vNIM claims it received")
    response_cid = get_cid_for_bytes(b"the exact response bytes")

    with caplog.at_level("ERROR"):
        handler.register_external_chat_transport(
            local_request_cid=local_request_cid,
            vnim_request_cid=vnim_request_cid,
            vnim_response_cid=response_cid,
            local_response_cid=response_cid,
            name="vNIM",
        )

    kinds = [kind for _, kind, _, _ in calls]
    assert kinds == ["external_request", "request_integrity_mismatch", "external_response"]
    assert calls[1][2:] == ([local_request_cid], [vnim_request_cid])
    assert "request_integrity_mismatch" in caplog.text


def test_response_mismatch_is_flagged_and_does_not_raise(sdk, monkeypatch, caplog):
    from eqty_sdk import get_cid_for_bytes

    input_cid = get_cid_for_bytes(b"input")
    output_cid = get_cid_for_bytes(b"output")
    handler, calls = _handler_with_deferred_run(monkeypatch, input_cid, output_cid)

    request_cid = get_cid_for_bytes(b"the exact request bytes")
    vnim_response_cid = get_cid_for_bytes(b"what vNIM claims it sent back")
    local_response_cid = get_cid_for_bytes(b"what this app actually received")

    with caplog.at_level("ERROR"):
        handler.register_external_chat_transport(
            local_request_cid=request_cid,
            vnim_request_cid=request_cid,
            vnim_response_cid=vnim_response_cid,
            local_response_cid=local_response_cid,
            name="vNIM",
        )

    kinds = [kind for _, kind, _, _ in calls]
    assert kinds == ["external_request", "response_integrity_mismatch", "external_response_unverified"]
    assert calls[1][2:] == ([vnim_response_cid], [local_response_cid])
    assert calls[2][2:] == ([local_response_cid], [output_cid])
    assert "response_integrity_mismatch" in caplog.text


def test_missing_local_response_cid_is_unverified_not_a_mismatch(sdk, monkeypatch, caplog):
    from eqty_sdk import get_cid_for_bytes

    input_cid = get_cid_for_bytes(b"input")
    output_cid = get_cid_for_bytes(b"output")
    handler, calls = _handler_with_deferred_run(monkeypatch, input_cid, output_cid)

    request_cid = get_cid_for_bytes(b"the exact request bytes")
    response_cid = get_cid_for_bytes(b"the exact response bytes")

    with caplog.at_level("WARNING"):
        handler.register_external_chat_transport(
            local_request_cid=request_cid,
            vnim_request_cid=request_cid,
            vnim_response_cid=response_cid,
            local_response_cid=None,
            name="vNIM",
        )

    kinds = [kind for _, kind, _, _ in calls]
    assert kinds == ["external_request", "external_response_unverified"]
    assert "response_not_independently_verified" in caplog.text
