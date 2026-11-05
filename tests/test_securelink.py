"""Run with python tests/test_securelink.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace
from unittest.mock import patch

import numpy as np
from cryptography.exceptions import InvalidTag

from securelink import (ACTIVE, BCH_ECC_BITS, FALLBACK, _bch_helper, _reconcile_bch,
                        accept_alice_offer, create_alice_offer, decrypt_payload,
                        derive_session_key, encrypt_payload, finalize_alice_session)


def paired_channels(seed=42):
    rng = np.random.default_rng(seed)
    frames, rows, columns = 32, 16, 32
    base = rng.normal(size=(rows, columns, 2)).astype(np.float32)
    motion = rng.normal(size=(rows, columns, 2)).astype(np.float32)
    time = np.sin(np.linspace(0, 4 * np.pi, frames, dtype=np.float32))[:, None, None, None]
    channel = base + 0.2 * time * motion
    alice = channel + rng.normal(scale=0.001, size=channel.shape).astype(np.float32)
    bob = channel + rng.normal(scale=0.001, size=channel.shape).astype(np.float32)
    return alice, bob


def main():
    source_bits = np.random.default_rng(1).integers(0, 2, 400, dtype=np.uint8).astype(bool)
    noisy_bits = source_bits.copy()
    noisy_bits[[0, 9, 18, 27, 36, 45]] ^= True
    reconciled, corrected = _reconcile_bch(noisy_bits, _bch_helper(source_bits))
    assert corrected == 6 and np.array_equal(reconciled, source_bits)
    uncorrectable = source_bits.copy()
    uncorrectable[[0, 1, 2, 3, 4, 5, 6]] ^= True
    try:
        reconciled, _ = _reconcile_bch(uncorrectable, _bch_helper(source_bits))
        assert not np.array_equal(reconciled, source_bits)
    except ValueError:
        pass

    secret = b"authenticated-device-secret".ljust(32, b"!")
    nonce = b"session-nonce-001"
    alice, bob = paired_channels()
    active = derive_session_key(alice, bob, secret, nonce, b"vehicle-rsu")
    assert active.status == ACTIVE and len(active.key) == 32
    assert active.retained_bits >= 256 and active.confirmation_tag
    assert active.helper_leakage_bits == BCH_ECC_BITS
    repeated = derive_session_key(alice, bob, secret, nonce, b"vehicle-rsu")
    assert repeated.key == active.key and repeated.confirmation_tag == active.confirmation_tag
    assert derive_session_key(alice, bob, secret, b"session-nonce-002", b"vehicle-rsu").key != active.key

    seen_nonces = set()
    offer, alice_session = create_alice_offer(alice, secret, nonce, b"vehicle-rsu")
    with patch("securelink._local_observation", return_value=(np.zeros(offer.feature_count), 1.0)):
        try:
            accept_alice_offer(bob, offer, secret, nonce, b"vehicle-rsu", set())
            raise AssertionError("weak Bob measurements were accepted")
        except ValueError as error:
            assert str(error) == "insufficient jointly reliable key bits"
    reply, bob_session = accept_alice_offer(bob, offer, secret, nonce, b"vehicle-rsu", seen_nonces)
    alice_key = finalize_alice_session(alice_session, reply)
    assert alice_key == bob_session.key and offer.nonce in seen_nonces
    payload_nonce = b"payload-0001"
    ciphertext = encrypt_payload(alice_key, payload_nonce, b"brake status", b"vehicle-rsu")
    assert decrypt_payload(bob_session.key, payload_nonce, ciphertext, b"vehicle-rsu") == b"brake status"

    try:
        accept_alice_offer(bob, offer, secret, nonce, b"vehicle-rsu", seen_nonces)
        raise AssertionError("replayed offer was accepted")
    except ValueError as error:
        assert str(error) == "replayed session nonce"
    tampered_offer = replace(offer, helper_data=bytes([offer.helper_data[0] ^ 1]) + offer.helper_data[1:])
    try:
        accept_alice_offer(bob, tampered_offer, secret, nonce, b"vehicle-rsu", set())
        raise AssertionError("tampered helper data was accepted")
    except ValueError as error:
        assert str(error) == "offer authentication failed"
    try:
        decrypt_payload(bob_session.key, payload_nonce, ciphertext[:-1] + bytes([ciphertext[-1] ^ 1]),
                        b"vehicle-rsu")
        raise AssertionError("tampered ciphertext was accepted")
    except InvalidTag:
        pass

    rng = np.random.default_rng(7)
    static = np.repeat(alice[:1], len(alice), axis=0)
    static_alice = static + rng.normal(scale=1e-5, size=static.shape).astype(np.float32)
    static_bob = static + rng.normal(scale=1e-5, size=static.shape).astype(np.float32)
    fallback = derive_session_key(static_alice, static_bob, secret, nonce, b"vehicle-rsu")
    assert fallback.status == FALLBACK and fallback.reason == "insufficient shared channel variation"
    assert len(fallback.key) == 32 and fallback.key != active.key

    unrelated = rng.normal(size=bob.shape).astype(np.float32)
    fallback = derive_session_key(alice, unrelated, secret, nonce, b"vehicle-rsu")
    assert fallback.status == FALLBACK and fallback.reason == "insufficient channel reciprocity"
    print("PASS: BCH reconciliation, authenticated handshake, replay rejection and AES-GCM")


if __name__ == "__main__":
    main()
