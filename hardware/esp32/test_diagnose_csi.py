import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from hardware.esp32.diagnose_csi import paired_phase_diagnostics, receiver_amplitude_mismatch, repeatability, temporal_correlation


if __name__ == "__main__":
    baseline = np.arange(1, 64, dtype=float)
    shared = np.stack([baseline + np.sin(index / 4) for index in range(32)])
    opposite = np.stack([baseline - np.sin(index / 4) for index in range(32)])
    assert temporal_correlation(shared, shared) > 0.99
    assert temporal_correlation(shared, opposite) < -0.99
    assert repeatability(shared)["median_adjacent_spectrum_correlation"] > 0.99
    assert repeatability(shared)["median_20_packet_relative_change"] > 0
    assert repeatability(shared)["median_adjacent_shape_change_after_gain_normalization"] >= 0
    gain_only = np.stack([baseline * (1 + 0.1 * np.sin(index)) for index in range(32)])
    assert repeatability(gain_only)["median_adjacent_relative_change"] > 0.01
    assert repeatability(gain_only)["median_adjacent_shape_change_after_gain_normalization"] < 1e-12
    positions = np.linspace(-1, 1, len(baseline))
    stats = receiver_amplitude_mismatch(np.stack([baseline] * 32),
                                        np.stack([baseline * 0.9 * (1 + 0.02 * positions)] * 32))
    assert all(0.9 <= value <= 0.92 for value in stats["bob_to_alice_gain_ratio_p05_p50_p95"])
    np.testing.assert_allclose(stats["absolute_log_amplitude_slope_p50_p95"], 0.02, atol=0.001)
    complex_alice = np.stack([baseline.astype(complex)] * 32)
    fixed_phase = paired_phase_diagnostics(complex_alice, complex_alice * 1j)
    assert fixed_phase["median_phase_invariant_spectrum_coherence"] > 0.999
    np.testing.assert_allclose(fixed_phase["paired_phase_offset_change_deg_p50_p95"], 0, atol=1e-8)
    drifting_bob = complex_alice * np.exp(1j * np.arange(32)[:, None] * np.pi / 2)
    drifting_phase = paired_phase_diagnostics(complex_alice, drifting_bob)
    assert drifting_phase["median_phase_invariant_spectrum_coherence"] > 0.999
    np.testing.assert_allclose(drifting_phase["paired_phase_offset_change_deg_p50_p95"], 90, atol=1e-8)
    print("PASS: CSI repeatability and paired-change diagnostics")
