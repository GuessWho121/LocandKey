"""Run SecureLink handshakes on matched hardware CSI without exporting keys."""

import argparse
from collections import Counter
import hmac
import json
import os
import time

import numpy as np

from securelink import (accept_alice_offer, create_alice_offer, decrypt_payload,
                        derive_session_key, encrypt_payload, finalize_alice_session)


def evaluate(path, rounds_per_session=4):
    with np.load(path) as capture:
        alice = capture["alice"]
        bob = capture["bob"]
    if alice.shape != bob.shape or alice.ndim != 4 or len(alice) < rounds_per_session:
        raise ValueError("Capture must contain matching Alice/Bob CSI rounds")
    secret = os.urandom(32)
    seen_nonces = set()
    results = []
    for start in range(0, len(alice) - rounds_per_session + 1, rounds_per_session):
        started = time.perf_counter()
        alice_window = alice[start:start + rounds_per_session]
        bob_window = bob[start:start + rounds_per_session]
        nonce = os.urandom(16)
        transcript = f"hardware-capture/{start}".encode()
        baseline = derive_session_key(alice_window, bob_window, secret, nonce, transcript)
        endpoint_status = "PHY_ACTIVE"
        reason = ""
        encrypted_payload_verified = False
        try:
            offer, alice_session = create_alice_offer(alice_window, secret, nonce, transcript)
            reply, bob_session = accept_alice_offer(
                bob_window, offer, secret, nonce, transcript, seen_nonces)
            alice_key = finalize_alice_session(alice_session, reply)
            assert hmac.compare_digest(alice_key, bob_session.key), "silent key mismatch"
            if baseline.status == "PHY_ACTIVE":
                payload_nonce = os.urandom(12)
                ciphertext = encrypt_payload(alice_key, payload_nonce, b"ESP32 capture verified", transcript)
                assert decrypt_payload(bob_session.key, payload_nonce, ciphertext, transcript) == b"ESP32 capture verified"
                encrypted_payload_verified = True
        except ValueError as error:
            endpoint_status = "REJECTED"
            reason = str(error)
        results.append({
            "round_start": start,
            "endpoint_status": endpoint_status,
            "endpoint_reason": reason,
            "host_policy": "PHY_ACTIVE" if encrypted_payload_verified else "FALLBACK",
            "encrypted_payload_verified": encrypted_payload_verified,
            "central_mode": baseline.status,
            "central_reason": baseline.reason,
            "reciprocal_correlation": baseline.reciprocal_correlation,
            "relative_temporal_variance": baseline.relative_temporal_variance,
            "retained_bits": baseline.retained_bits,
            "raw_disagreement_rate": baseline.raw_disagreement_rate,
            "corrected_bits": baseline.corrected_bits,
            "latency_ms": (time.perf_counter() - started) * 1000,
        })
    return {
        "capture": str(path),
        "rounds": len(alice),
        "sessions": len(results),
        "endpoint_outcomes": dict(Counter(row["endpoint_status"] for row in results)),
        "central_modes": dict(Counter(row["central_mode"] for row in results)),
        "host_policy_outcomes": dict(Counter(row["host_policy"] for row in results)),
        "unsafe_endpoint_activations_blocked": sum(
            row["endpoint_status"] == "PHY_ACTIVE" and row["host_policy"] != "PHY_ACTIVE"
            for row in results),
        "p95_host_latency_ms": float(np.percentile([row["latency_ms"] for row in results], 95)),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Check real ESP32 CSI with the SecureLink protocol.")
    parser.add_argument("capture")
    parser.add_argument("--rounds-per-session", type=int, default=4)
    args = parser.parse_args()
    if args.rounds_per_session < 4:
        parser.error("rounds-per-session must be at least four")
    print(json.dumps(evaluate(args.capture, args.rounds_per_session), indent=2))


if __name__ == "__main__":
    main()
