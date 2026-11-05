"""Run with python tests/test_securelink_product.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluate_securelink import aggregate, acceptance, make_session, run_handshake
from securelink_capture import match_packets, parse_packet, to_securelink_tensors


def packet(device, sequence, timestamp):
    return parse_packet(json.dumps({
        "device": device, "sequence": sequence, "timestamp_us": timestamp,
        "rssi_dbm": -48, "noise_floor_dbm": -94, "channel": 6, "bandwidth_mhz": 20,
        "csi": list(range(128)),
    }))


def main():
    rng = np.random.default_rng(42)
    alice, bob, _ = make_session(rng, 10, 16, 16, 32, (0.0,))
    nonce = b"product-test-001"
    result, offer, state = run_handshake(alice, bob, nonce, b"product-test")
    assert result["success"] and offer is not None and state is not None
    assert not result["silent_mismatch"] and not result["tamper_accepted"]
    assert not result["replay_accepted"] and result["retained_bits"] >= 256

    rows = [{"mode": "raw", "snr_db": 10.0, "eve_correlation": 0.0, "trial": index,
             **result, "eve_confirmed": 0} for index in range(2)]
    metrics = aggregate(rows)
    assert acceptance(metrics, 10.0)["pass"]

    alice_packets = [packet("alice", index, index * 1_000) for index in range(32)]
    bob_packets = [packet("bob", index, index * 1_000 + 100) for index in range(32)]
    pairs = match_packets(alice_packets, bob_packets)
    alice_tensor, bob_tensor = to_securelink_tensors(pairs)
    assert alice_tensor.shape == bob_tensor.shape == (4, 8, 64, 2)
    alice_with_rtt = parse_packet(json.dumps({
        "device": "alice", "sequence": 1, "timestamp_us": 1_000_000,
        "round_trip_us": 900, "rssi_dbm": -48, "noise_floor_dbm": -94,
        "channel": 6, "bandwidth_mhz": 20, "csi": list(range(128)),
    }))
    assert len(match_packets([alice_with_rtt], [packet("bob", 1, 50)])) == 1
    assert not match_packets([parse_packet(json.dumps({
        "device": "alice", "sequence": 1, "timestamp_us": 1_000_000,
        "round_trip_us": 5_001, "rssi_dbm": -48, "noise_floor_dbm": -94,
        "channel": 6, "bandwidth_mhz": 20, "csi": list(range(128)),
    }))], [packet("bob", 1, 50)])
    alice_capture = parse_packet(json.dumps({
        "device": "alice", "capture_id": 10, "sequence": 1, "timestamp_us": 1000,
        "round_trip_us": 900, "rssi_dbm": -48, "noise_floor_dbm": -94,
        "channel": 6, "bandwidth_mhz": 20, "csi": list(range(128)),
    }))
    bob_capture = parse_packet(json.dumps({
        "device": "bob", "capture_id": 11, "sequence": 1, "timestamp_us": 1100,
        "rssi_dbm": -48, "noise_floor_dbm": -94,
        "channel": 6, "bandwidth_mhz": 20, "csi": list(range(128)),
    }))
    assert not match_packets([alice_capture], [bob_capture])
    assert len(match_packets([alice_capture], [parse_packet(json.dumps({
        "device": "bob", "capture_id": 10, "sequence": 1, "timestamp_us": 1100,
        "rssi_dbm": -48, "noise_floor_dbm": -94,
        "channel": 6, "bandwidth_mhz": 20, "csi": list(range(128)),
    }))])) == 1
    try:
        parse_packet(json.dumps({
            "device": "bob", "capture_id": -1, "sequence": 1, "timestamp_us": 1100,
            "rssi_dbm": -48, "noise_floor_dbm": -94,
            "channel": 6, "bandwidth_mhz": 20, "csi": list(range(128)),
        }))
        raise AssertionError("Invalid capture ID accepted")
    except ValueError:
        pass
    print("PASS: evaluator acceptance and ESP32 reciprocal capture conversion")


if __name__ == "__main__":
    main()
