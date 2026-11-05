"""Parse paired ESP32 CSI JSONL captures into SecureLink tensors."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CSIPacket:
    device: str
    sequence: int
    timestamp_us: int
    rssi_dbm: int
    noise_floor_dbm: int
    channel: int
    bandwidth_mhz: int
    csi: np.ndarray
    round_trip_us: int | None = None
    capture_id: int | None = None


def parse_packet(line):
    record = json.loads(line)
    required = {"device", "sequence", "timestamp_us", "rssi_dbm", "noise_floor_dbm",
                "channel", "bandwidth_mhz", "csi"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Missing CSI fields: {sorted(missing)}")
    values = np.asarray(record["csi"], dtype=np.float32)
    if values.ndim != 1 or len(values) < 2 or len(values) % 2 or not np.isfinite(values).all():
        raise ValueError("csi must be a finite interleaved real/imaginary list")
    if int(record["sequence"]) < 0 or int(record["timestamp_us"]) < 0:
        raise ValueError("sequence and timestamp_us must be nonnegative")
    round_trip_us = record.get("round_trip_us")
    if round_trip_us is not None and int(round_trip_us) < 0:
        raise ValueError("round_trip_us must be nonnegative")
    capture_id = record.get("capture_id")
    if capture_id is not None and not 0 <= int(capture_id) <= 0xFFFFFFFF:
        raise ValueError("capture_id must be a 32-bit unsigned integer")
    return CSIPacket(str(record["device"]), int(record["sequence"]), int(record["timestamp_us"]),
                     int(record["rssi_dbm"]), int(record["noise_floor_dbm"]),
                     int(record["channel"]), int(record["bandwidth_mhz"]), values.reshape(-1, 2),
                     None if round_trip_us is None else int(round_trip_us),
                     None if capture_id is None else int(capture_id))


def read_packets(path):
    packets = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    packets.append(parse_packet(line))
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not packets:
        raise ValueError(f"No CSI packets in {path}")
    return packets


def match_packets(alice_packets, bob_packets, max_delta_us=5_000):
    bob_by_sequence = {(packet.capture_id, packet.sequence): packet for packet in bob_packets}
    pairs = []
    for alice in alice_packets:
        bob = bob_by_sequence.get((alice.capture_id, alice.sequence))
        timing_us = (alice.round_trip_us if alice.round_trip_us is not None else
                     abs(alice.timestamp_us - bob.timestamp_us) if bob is not None else max_delta_us + 1)
        if (bob is not None and timing_us <= max_delta_us
                and alice.channel == bob.channel and alice.bandwidth_mhz == bob.bandwidth_mhz
                and alice.csi.shape == bob.csi.shape):
            pairs.append((alice, bob))
    return pairs


def to_securelink_tensors(pairs, packets_per_round=8):
    if packets_per_round <= 0:
        raise ValueError("packets_per_round must be positive")
    rounds = len(pairs) // packets_per_round
    if rounds < 4:
        raise ValueError("At least four complete reciprocal CSI rounds are required")
    pairs = pairs[:rounds * packets_per_round]
    alice = np.stack([pair[0].csi for pair in pairs])
    bob = np.stack([pair[1].csi for pair in pairs])
    subcarriers = alice.shape[1]
    shape = (rounds, packets_per_round, subcarriers, 2)
    return alice.reshape(shape), bob.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="Convert reciprocal ESP32 CSI JSONL to NPZ.")
    parser.add_argument("--alice-jsonl", required=True)
    parser.add_argument("--bob-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-delta-us", type=int, default=5_000)
    parser.add_argument("--packets-per-round", type=int, default=8)
    args = parser.parse_args()
    alice_packets = read_packets(args.alice_jsonl)
    bob_packets = read_packets(args.bob_jsonl)
    pairs = match_packets(alice_packets, bob_packets, args.max_delta_us)
    alice, bob = to_securelink_tensors(pairs, args.packets_per_round)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, alice=alice, bob=bob)
    print(json.dumps({"matched_packets": len(pairs), "tensor_shape": list(alice.shape),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
