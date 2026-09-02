from precourier.webhooks import sign_payload, verify_payload


def test_webhook_signatures_are_exact_and_tamper_evident():
    body = b'{"event":"order.released","order_id":"ord_1"}'
    signature = sign_payload("secret", body)
    assert verify_payload("secret", body, signature)
    assert verify_payload("secret", body, "sha256=" + signature)
    assert not verify_payload("secret", body + b" ", signature)
    assert not verify_payload("wrong", body, signature)
