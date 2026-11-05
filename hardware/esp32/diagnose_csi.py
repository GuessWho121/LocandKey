"""Report radio metadata and CSI repeatability for a paired ESP32 capture."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from securelink_capture import match_packets, read_packets


METADATA = (
    "channel", "bandwidth_mhz", "signal_mode", "rate", "mcs",
    "secondary_channel", "aggregation", "antenna", "rx_state",
    "first_word_invalid", "csi_length", "rssi_dbm", "noise_floor_dbm",
)
LINK_SETTINGS = ("channel", "bandwidth_mhz", "signal_mode", "rate", "mcs", "secondary_channel")


def temporal_correlation(alice, bob):
    alice_changes = alice - alice.mean(axis=0)
    bob_changes = bob - bob.mean(axis=0)
    denominator = np.linalg.norm(alice_changes) * np.linalg.norm(bob_changes)
    return float(np.sum(alice_changes * bob_changes) / denominator) if denominator else 0.0


def repeatability(magnitudes):
    def change_at_lag(lag):
        differences = np.linalg.norm(magnitudes[lag:] - magnitudes[:-lag], axis=1)
        baseline = np.linalg.norm(magnitudes[:-lag], axis=1)
        return float(np.median(differences / np.maximum(baseline, 1e-12)))

    centered = magnitudes - magnitudes.mean(axis=1, keepdims=True)
    adjacent = centered[1:] * centered[:-1]
    denominator = np.linalg.norm(centered[1:], axis=1) * np.linalg.norm(centered[:-1], axis=1)
    shape_correlation = np.sum(adjacent, axis=1) / np.maximum(denominator, 1e-12)
    packet_gain = np.sqrt(np.mean(magnitudes ** 2, axis=1))
    normalized = magnitudes / np.maximum(packet_gain[:, None], 1e-12)
    return {
        "median_adjacent_spectrum_correlation": float(np.median(shape_correlation)),
        "median_adjacent_relative_change": change_at_lag(1),
        "median_20_packet_relative_change": change_at_lag(20) if len(magnitudes) > 20 else None,
        "median_adjacent_gain_change": float(np.median(np.abs(np.diff(packet_gain)) /
                                                         np.maximum(packet_gain[:-1], 1e-12))),
        "median_adjacent_shape_change_after_gain_normalization": float(np.median(
            np.linalg.norm(np.diff(normalized, axis=0), axis=1) /
            np.maximum(np.linalg.norm(normalized[:-1], axis=1), 1e-12))),
    }


def receiver_amplitude_mismatch(alice, bob):
    alice_gain = np.sqrt(np.mean(alice ** 2, axis=1))
    bob_gain = np.sqrt(np.mean(bob ** 2, axis=1))
    gain_ratio = bob_gain / np.maximum(alice_gain, 1e-12)
    positions = np.linspace(-1, 1, alice.shape[1])
    shape_ratio = (np.log(np.maximum(bob / np.maximum(bob_gain[:, None], 1e-12), 1e-3))
                   - np.log(np.maximum(alice / np.maximum(alice_gain[:, None], 1e-12), 1e-3)))
    slope = shape_ratio @ positions / (positions @ positions)
    return {
        "bob_to_alice_gain_ratio_p05_p50_p95": np.percentile(gain_ratio, [5, 50, 95]).tolist(),
        "absolute_log_amplitude_slope_p50_p95": np.percentile(np.abs(slope), [50, 95]).tolist(),
    }


def paired_phase_diagnostics(alice, bob):
    cross = np.sum(alice * np.conj(bob), axis=1)
    coherence = np.abs(cross) / np.maximum(np.linalg.norm(alice, axis=1) *
                                            np.linalg.norm(bob, axis=1), 1e-12)
    phase_change = np.degrees(np.abs(np.angle(cross[1:] * np.conj(cross[:-1]))))
    return {
        "median_phase_invariant_spectrum_coherence": float(np.median(coherence)),
        "paired_phase_offset_change_deg_p50_p95": np.percentile(phase_change, [50, 95]).tolist(),
    }


def read_metadata(path):
    return {
        (record.get("capture_id"), record["sequence"]): record
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
        for record in (json.loads(line),)
    }


def diagnose(alice_path, bob_path):
    alice_packets = read_packets(alice_path)
    bob_packets = read_packets(bob_path)
    pairs = match_packets(alice_packets, bob_packets)
    if len(pairs) < 21:
        raise ValueError("At least 21 matched CSI packets are required")
    alice_metadata = read_metadata(alice_path)
    bob_metadata = read_metadata(bob_path)
    magnitudes = [
        np.stack([np.linalg.norm(packet.csi[2:], axis=1) for packet in side])
        for side in ([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    ]
    complex_csi = [
        np.stack([packet.csi[2:, 0] + 1j * packet.csi[2:, 1] for packet in side])
        for side in ([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    ]
    metadata_counts = {}
    for name, rows in (("alice", alice_metadata), ("bob", bob_metadata)):
        metadata_counts[name] = {
            field: dict(sorted(Counter(str(row.get(field, "missing")) for row in rows.values()).items()))
            for field in METADATA
        }
    mismatches = {
        field: sum(alice_metadata[(alice.capture_id, alice.sequence)].get(field) !=
                   bob_metadata[(bob.capture_id, bob.sequence)].get(field)
                   for alice, bob in pairs)
        for field in LINK_SETTINGS
    }
    return {
        "alice_packets": len(alice_packets),
        "bob_packets": len(bob_packets),
        "matched_packets": len(pairs),
        "p95_round_trip_ms": float(np.percentile([alice.round_trip_us for alice, _ in pairs], 95) / 1000),
        "metadata_counts": metadata_counts,
        "paired_setting_mismatches": mismatches,
        "alice_repeatability": repeatability(magnitudes[0]),
        "bob_repeatability": repeatability(magnitudes[1]),
        "paired_receiver_amplitude": receiver_amplitude_mismatch(*magnitudes),
        "paired_phase": paired_phase_diagnostics(*complex_csi),
        "paired_temporal_correlation": temporal_correlation(*magnitudes),
        "paired_temporal_correlation_after_gain_normalization": temporal_correlation(*[
            magnitude / np.maximum(np.sqrt(np.mean(magnitude ** 2, axis=1))[:, None], 1e-12)
            for magnitude in magnitudes
        ]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alice_jsonl")
    parser.add_argument("bob_jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = json.dumps(diagnose(args.alice_jsonl, args.bob_jsonl), indent=2)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    print(report)
