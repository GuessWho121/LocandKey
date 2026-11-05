"""Fail-safe SecureLink session-key derivation from paired CSI observations."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import struct

import bchlib
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ACTIVE = "PHY_ACTIVE"
FALLBACK = "FALLBACK"
BCH_T = 6
BCH_M = 9
_BCH_PARAMETERS = bchlib.BCH(BCH_T, m=BCH_M)
BCH_ECC_BITS = _BCH_PARAMETERS.ecc_bits
BCH_ECC_BYTES = _BCH_PARAMETERS.ecc_bytes
BCH_MAX_DATA_BITS = ((_BCH_PARAMETERS.n - BCH_ECC_BITS) // 8) * 8


@dataclass(frozen=True)
class SecureLinkResult:
    status: str
    key: bytes
    reason: str
    reciprocal_correlation: float
    relative_temporal_variance: float
    retained_bits: int
    bit_balance: float
    raw_disagreement_rate: float
    corrected_bits: int
    helper_leakage_bits: int
    reliability_digest: str
    confirmation_tag: str


@dataclass(frozen=True)
class SecureLinkOffer:
    nonce: bytes
    transcript_digest: bytes
    feature_count: int
    reliability_mask: bytes
    helper_data: bytes
    auth_tag: bytes


@dataclass(frozen=True)
class SecureLinkReply:
    offer_digest: bytes
    confirmation_tag: bytes


@dataclass(frozen=True)
class AliceSession:
    key: bytes
    offer_digest: bytes


@dataclass(frozen=True)
class BobSession:
    key: bytes
    correction_rate: float
    corrected_bits: int
    helper_leakage_bits: int


def hkdf_sha384(material, salt, info, length=32):
    if length <= 0 or length > 255 * hashlib.sha384().digest_size:
        raise ValueError("Invalid HKDF output length")
    extract = hmac.new(salt, material, hashlib.sha384).digest()
    output, block = bytearray(), b""
    for counter in range(1, (length + hashlib.sha384().digest_size - 1) // hashlib.sha384().digest_size + 1):
        block = hmac.new(extract, block + info + bytes([counter]), hashlib.sha384).digest()
        output.extend(block)
    return bytes(output[:length])


def _validate_pair(alice, bob):
    alice = np.asarray(alice, dtype=np.float32)
    bob = np.asarray(bob, dtype=np.float32)
    if alice.shape != bob.shape or alice.ndim < 3 or alice.shape[0] < 4 or alice.shape[-1] != 2:
        raise ValueError("Alice and Bob CSI must share shape (frames, ..., 2) with at least four frames")
    if not np.isfinite(alice).all() or not np.isfinite(bob).all():
        raise ValueError("CSI must contain only finite values")
    return alice, bob


def _channel_metrics(alice, bob):
    alice_mag = np.linalg.norm(alice, axis=-1)
    bob_mag = np.linalg.norm(bob, axis=-1)
    alice_dynamic = alice_mag - alice_mag.mean(axis=0, keepdims=True)
    bob_dynamic = bob_mag - bob_mag.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(alice_dynamic) * np.linalg.norm(bob_dynamic)
    correlation = float(np.sum(alice_dynamic * bob_dynamic) / denominator) if denominator else 0.0
    temporal_variance = min(float(np.mean(alice_dynamic ** 2)), float(np.mean(bob_dynamic ** 2)))
    channel_power = min(float(np.mean(alice_mag ** 2)), float(np.mean(bob_mag ** 2)))
    relative_variance = temporal_variance / max(channel_power, np.finfo(np.float32).tiny)
    return alice_mag, bob_mag, correlation, relative_variance


def _quantize(alice_mag, bob_mag, guard_ratio):
    alice_feature = _normalized_feature(alice_mag)
    bob_feature = _normalized_feature(bob_mag)
    reliable = (np.abs(alice_feature) >= guard_ratio) & (np.abs(bob_feature) >= guard_ratio)
    return alice_feature[reliable] >= 0, bob_feature[reliable] >= 0, np.packbits(reliable).tobytes()


def _normalized_feature(magnitude):
    feature = magnitude.mean(axis=0).reshape(-1)
    centered = feature - np.median(feature)
    return centered / max(float(np.std(centered)), np.finfo(np.float32).eps)


def _bch_helper(bits):
    bits = np.asarray(bits, dtype=np.bool_).reshape(-1)
    if not len(bits) or len(bits) % 8 or len(bits) > BCH_MAX_DATA_BITS:
        raise ValueError("BCH input must be whole bytes within the configured code size")
    return bchlib.BCH(BCH_T, m=BCH_M).encode(np.packbits(bits).tobytes())


def _reconcile_bch(bits, alice_helper):
    bits = np.asarray(bits, dtype=np.bool_).reshape(-1)
    if not len(bits) or len(bits) % 8 or len(bits) > BCH_MAX_DATA_BITS:
        raise ValueError("BCH input must be whole bytes within the configured code size")
    if len(alice_helper) != BCH_ECC_BYTES:
        raise ValueError("Invalid BCH helper length")
    bch = bchlib.BCH(BCH_T, m=BCH_M)
    data = bytearray(np.packbits(bits).tobytes())
    helper = bytearray(alice_helper)
    corrected_bits = bch.decode(data=data, recv_ecc=helper)
    if corrected_bits < 0:
        raise ValueError("BCH reconciliation failed")
    bch.correct(data=data, ecc=helper)
    corrected = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(bool)[:len(bits)]
    return corrected, corrected_bits


def _offer_payload(offer):
    header = struct.pack(">HIII", len(offer.nonce), offer.feature_count,
                         len(offer.reliability_mask), len(offer.helper_data))
    return (b"securelink/offer/v2" + header + offer.nonce + offer.transcript_digest
            + offer.reliability_mask + offer.helper_data)


def _offer_digest(offer):
    return hashlib.sha384(_offer_payload(offer) + offer.auth_tag).digest()


def _local_observation(channel):
    channel, _ = _validate_pair(channel, channel)
    magnitude = np.linalg.norm(channel, axis=-1)
    dynamic = magnitude - magnitude.mean(axis=0, keepdims=True)
    variance = float(np.mean(dynamic ** 2))
    power = float(np.mean(magnitude ** 2))
    return _normalized_feature(magnitude), variance / max(power, np.finfo(np.float32).tiny)


def create_alice_offer(alice, authenticated_secret, nonce, transcript=b"", *,
                       min_relative_variance=1e-4, guard_ratio=0.25, min_bits=256):
    """Create Alice's authenticated public reconciliation message and private session state."""
    if len(authenticated_secret) < 32 or len(nonce) < 16:
        raise ValueError("Authenticated secret and nonce must be at least 32 and 16 bytes")
    feature, relative_variance = _local_observation(alice)
    reliable_indices = np.flatnonzero(np.abs(feature) >= guard_ratio)
    usable = min(len(reliable_indices) // 8 * 8, BCH_MAX_DATA_BITS)
    reliable_indices = reliable_indices[:usable]
    bits = feature[reliable_indices] >= 0
    balance = float(bits.mean()) if usable else 0.0
    if relative_variance < min_relative_variance:
        raise ValueError("insufficient local channel variation")
    if usable < min_bits:
        raise ValueError("insufficient reliable key bits")
    if not 0.25 <= balance <= 0.75:
        raise ValueError("channel bits failed balance gate")

    reliability = np.zeros(len(feature), dtype=np.bool_)
    reliability[reliable_indices] = True
    helper_data = _bch_helper(bits)
    unsigned = SecureLinkOffer(nonce, hashlib.sha384(transcript).digest(), len(feature),
                               np.packbits(reliability).tobytes(), helper_data, b"")
    auth_tag = hmac.new(authenticated_secret, _offer_payload(unsigned), hashlib.sha384).digest()
    offer = SecureLinkOffer(unsigned.nonce, unsigned.transcript_digest, unsigned.feature_count,
                            unsigned.reliability_mask, unsigned.helper_data, auth_tag)
    offer_digest = _offer_digest(offer)
    key = hkdf_sha384(authenticated_secret + np.packbits(bits).tobytes(), nonce,
                      b"securelink/channel/v4" + offer_digest)
    return offer, AliceSession(key, offer_digest)


def accept_alice_offer(bob, offer, authenticated_secret, expected_nonce, transcript,
                       seen_nonces, *, min_relative_variance=1e-4,
                       guard_ratio=0.25, min_bits=256):
    """Authenticate Alice's offer, reject replay, reconcile Bob's bits, and confirm the key."""
    if len(authenticated_secret) < 32 or len(expected_nonce) < 16:
        raise ValueError("Authenticated secret and nonce must be at least 32 and 16 bytes")
    if not hmac.compare_digest(offer.nonce, expected_nonce):
        raise ValueError("unexpected session nonce")
    if not hmac.compare_digest(offer.transcript_digest, hashlib.sha384(transcript).digest()):
        raise ValueError("transcript mismatch")
    expected_tag = hmac.new(authenticated_secret, _offer_payload(offer), hashlib.sha384).digest()
    if not hmac.compare_digest(offer.auth_tag, expected_tag):
        raise ValueError("offer authentication failed")
    if offer.nonce in seen_nonces:
        raise ValueError("replayed session nonce")

    feature, relative_variance = _local_observation(bob)
    if relative_variance < min_relative_variance:
        raise ValueError("insufficient local channel variation")
    if offer.feature_count != len(feature):
        raise ValueError("channel feature shape mismatch")
    expected_mask_bytes = (offer.feature_count + 7) // 8
    if len(offer.reliability_mask) != expected_mask_bytes:
        raise ValueError("invalid reliability mask length")
    reliability = np.unpackbits(np.frombuffer(offer.reliability_mask, dtype=np.uint8),
                                count=offer.feature_count).astype(bool)
    if np.count_nonzero(np.abs(feature[reliability]) >= guard_ratio) < min_bits:
        raise ValueError("insufficient jointly reliable key bits")
    bob_bits = feature[reliability] >= 0
    if len(bob_bits) % 8 or len(bob_bits) > BCH_MAX_DATA_BITS:
        raise ValueError("invalid reconciliation data length")
    seen_nonces.add(offer.nonce)

    reconciled_bits, corrected_bits = _reconcile_bch(bob_bits, offer.helper_data)
    correction_rate = float(np.mean(bob_bits != reconciled_bits)) if len(bob_bits) else 0.0
    offer_digest = _offer_digest(offer)
    key = hkdf_sha384(authenticated_secret + np.packbits(reconciled_bits).tobytes(), offer.nonce,
                      b"securelink/channel/v4" + offer_digest)
    confirmation = hmac.new(key, b"securelink/bob-confirm/v2" + offer_digest,
                            hashlib.sha384).digest()
    reply = SecureLinkReply(offer_digest, confirmation)
    return reply, BobSession(key, correction_rate, corrected_bits, BCH_ECC_BITS)


def finalize_alice_session(session, reply):
    """Verify Bob's key confirmation before Alice uses the session key."""
    if not hmac.compare_digest(session.offer_digest, reply.offer_digest):
        raise ValueError("reply belongs to another offer")
    expected = hmac.new(session.key, b"securelink/bob-confirm/v2" + session.offer_digest,
                        hashlib.sha384).digest()
    if not hmac.compare_digest(reply.confirmation_tag, expected):
        raise ValueError("Bob key confirmation failed")
    return session.key


def encrypt_payload(key, nonce, plaintext, associated_data=b""):
    if len(key) != 32 or len(nonce) != 12:
        raise ValueError("AES-256-GCM requires a 32-byte key and 12-byte nonce")
    return AESGCM(key).encrypt(nonce, plaintext, associated_data)


def decrypt_payload(key, nonce, ciphertext, associated_data=b""):
    if len(key) != 32 or len(nonce) != 12:
        raise ValueError("AES-256-GCM requires a 32-byte key and 12-byte nonce")
    return AESGCM(key).decrypt(nonce, ciphertext, associated_data)


def derive_session_key(alice, bob, authenticated_secret, nonce, transcript=b"", *,
                       min_correlation=0.8, min_relative_variance=1e-4,
                       guard_ratio=0.25, min_bits=256):
    """Return a channel-bound key when gates pass, otherwise an authenticated fallback key."""
    if len(authenticated_secret) < 32 or len(nonce) < 16:
        raise ValueError("Authenticated secret and nonce must be at least 32 and 16 bytes")
    alice, bob = _validate_pair(alice, bob)
    alice_mag, bob_mag, correlation, relative_variance = _channel_metrics(alice, bob)
    alice_bits, bob_bits, reliability_mask = _quantize(alice_mag, bob_mag, guard_ratio)
    reliability_digest = hashlib.sha256(reliability_mask).hexdigest()
    usable = min(len(alice_bits) // 8 * 8, BCH_MAX_DATA_BITS)
    alice_bits, bob_bits = alice_bits[:usable], bob_bits[:usable]
    helper_data = _bch_helper(alice_bits) if usable else b""
    if usable:
        try:
            reconciled_bob_bits, corrected_bits = _reconcile_bch(bob_bits, helper_data)
        except ValueError:
            reconciled_bob_bits, corrected_bits = bob_bits, 0
    else:
        reconciled_bob_bits, corrected_bits = bob_bits, 0
    raw_disagreement_rate = float(np.mean(alice_bits != bob_bits)) if usable else 0.0
    helper_leakage_bits = BCH_ECC_BITS if helper_data else 0
    balance = float(alice_bits.mean()) if len(alice_bits) else 0.0

    reason = "channel gates passed"
    if relative_variance < min_relative_variance:
        reason = "insufficient shared channel variation"
    elif correlation < min_correlation:
        reason = "insufficient channel reciprocity"
    elif len(alice_bits) < min_bits:
        reason = "insufficient reliable key bits"
    elif not 0.25 <= balance <= 0.75:
        reason = "channel bits failed balance gate"
    elif not np.array_equal(alice_bits, reconciled_bob_bits):
        reason = "BCH reconciliation failed"

    if reason == "channel gates passed":
        alice_material = np.packbits(alice_bits).tobytes()
        bob_material = np.packbits(reconciled_bob_bits).tobytes()
        context = (b"securelink/channel/v2" + hashlib.sha384(transcript).digest()
                   + hashlib.sha256(reliability_mask).digest()
                   + hashlib.sha256(helper_data).digest())
        alice_key = hkdf_sha384(authenticated_secret + alice_material, nonce, context)
        bob_key = hkdf_sha384(authenticated_secret + bob_material, nonce, context)
        if not hmac.compare_digest(alice_key, bob_key):
            reason = "key confirmation failed"
        else:
            tag = hmac.new(alice_key, b"securelink/confirm/v2" + transcript, hashlib.sha384).hexdigest()
            return SecureLinkResult(ACTIVE, alice_key, reason, correlation, relative_variance,
                                    len(alice_bits), balance, raw_disagreement_rate, corrected_bits,
                                    helper_leakage_bits, reliability_digest, tag)

    fallback = hkdf_sha384(authenticated_secret, nonce, b"securelink/fallback/v1" + transcript)
    tag = hmac.new(fallback, b"securelink/confirm/v1" + transcript, hashlib.sha384).hexdigest()
    return SecureLinkResult(FALLBACK, fallback, reason, correlation, relative_variance,
                            len(alice_bits), balance, raw_disagreement_rate, corrected_bits,
                            helper_leakage_bits, reliability_digest, tag)


def _demo(seed):
    rng = np.random.default_rng(seed)
    frames, rows, columns = 32, 16, 32
    base = rng.normal(size=(rows, columns, 2)).astype(np.float32)
    motion = rng.normal(size=(rows, columns, 2)).astype(np.float32)
    time = np.sin(np.linspace(0, 4 * np.pi, frames, dtype=np.float32))[:, None, None, None]
    channel = base + 0.2 * time * motion
    alice = channel + rng.normal(scale=0.001, size=channel.shape).astype(np.float32)
    bob = channel + rng.normal(scale=0.001, size=channel.shape).astype(np.float32)
    secret = b"demo-authenticated-secret".ljust(32, b"!")
    nonce = b"demo-session-nonce"
    transcript = b"vehicle-rsu-demo"
    offer, alice_session = create_alice_offer(alice, secret, nonce, transcript)
    reply, bob_session = accept_alice_offer(bob, offer, secret, nonce, transcript, set())
    key = finalize_alice_session(alice_session, reply)
    # ponytail: one deterministic nonce is safe for this one-message demo; production needs a persistent counter.
    payload_nonce = hashlib.sha256(nonce + b"payload-1").digest()[:12]
    ciphertext = encrypt_payload(key, payload_nonce, b"secure vehicle message", transcript)
    recovered = decrypt_payload(bob_session.key, payload_nonce, ciphertext, transcript)
    report = asdict(bob_session)
    report.update({"status": ACTIVE, "key": hashlib.sha256(key).hexdigest()[:16],
                   "offer_authenticated": True, "payload_verified": recovered.decode("ascii"),
                   "ciphertext_bytes": len(ciphertext)})
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run the SecureLink paired-CSI product demo.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.demo:
        parser.error("choose --demo")
    _demo(args.seed)


if __name__ == "__main__":
    main()
