"""Verification and leakage tests for physically consistent paired Wi-Fi IoT channel simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from wifi_channel_sim import (
    SimConfig,
    ReceiverImpairment,
    generate_multipath_profile,
    evaluate_underlying_channel,
    apply_receiver_impairment,
    simulate_paired_iot_session,
    NULL_MASK,
    DC_INDEX,
)


def test_exact_static_reciprocity():
    """In static mode (Doppler=0) without hardware mismatch, channel is identical across packet times."""
    rng = np.random.default_rng(101)
    config = SimConfig(max_doppler_hz=0.0, num_paths=6)
    profile = generate_multipath_profile(config, rng)
    phases = rng.uniform(0, 2 * np.pi, size=config.num_paths)
    t_alice = np.array([0.0, 0.010, 0.020])
    t_bob = t_alice + 0.002  # 2 ms later

    h_a = evaluate_underlying_channel(profile, config, t_alice, phases)
    h_b = evaluate_underlying_channel(profile, config, t_bob, phases)

    max_diff = np.max(np.abs(h_a - h_b))
    assert max_diff < 1e-6, f"Static channel varied across time: diff={max_diff}"


def test_doppler_temporal_evolution():
    """Under motion (Doppler > 0), the channel evolves over time."""
    rng = np.random.default_rng(202)
    config = SimConfig(max_doppler_hz=20.0, num_paths=8)
    profile = generate_multipath_profile(config, rng)
    phases = rng.uniform(0, 2 * np.pi, size=config.num_paths)
    times = np.array([0.0, 0.5])  # 500 ms apart

    h = evaluate_underlying_channel(profile, config, times, phases)
    # Correlation between t=0 and t=500ms
    c0, c1 = h[0], h[1]
    norm_corr = np.abs(np.vdot(c0, c1)) / (np.linalg.norm(c0) * np.linalg.norm(c1))
    assert norm_corr < 0.95, f"Expected decorrelation over 500ms at 20 Hz Doppler, got {norm_corr:.3f}"


def test_null_masking():
    """DC and guard subcarriers must be strictly zeroed out."""
    rng = np.random.default_rng(303)
    config = SimConfig()
    profile = generate_multipath_profile(config, rng)
    phases = rng.uniform(0, 2 * np.pi, size=config.num_paths)
    h = evaluate_underlying_channel(profile, config, np.array([0.0]), phases)
    h_obs = apply_receiver_impairment(h, np.array([0.0]), ReceiverImpairment(), mask_nulls=True)

    assert np.all(h_obs[0, NULL_MASK] == 0.0)
    assert h_obs[0, DC_INDEX] == 0.0


def test_session_shapes_and_eve_isolation():
    """Test full session simulation shapes and verify Eve receives no shared statistics."""
    rng = np.random.default_rng(404)
    # With Doppler motion, temporal variation occurs and independent Eve decorrelates.
    config = SimConfig(subcarriers=64, max_doppler_hz=10.0, num_paths=8)
    rounds, packets = 4, 8
    alice, bob, eves = simulate_paired_iot_session(
        config,
        packets_per_round=packets,
        num_rounds=rounds,
        snr_db=20.0,
        eve_correlations=(0.0, 0.5, 0.9),
        rng=rng,
    )

    expected_shape = (rounds, packets, 64, 2)
    assert alice.shape == expected_shape, f"Alice shape {alice.shape} != {expected_shape}"
    assert bob.shape == expected_shape, f"Bob shape {bob.shape} != {expected_shape}"
    assert len(eves) == 3

    # Check finite numbers
    assert np.isfinite(alice).all()
    assert np.isfinite(bob).all()
    for e in eves:
        assert e.shape == expected_shape
        assert np.isfinite(e).all()

    # Zero-correlation Eve temporal variations should have low correlation with Alice
    a_changes = alice - alice.mean(axis=(0, 1), keepdims=True)
    e0_changes = eves[0] - eves[0].mean(axis=(0, 1), keepdims=True)
    denom = np.linalg.norm(a_changes) * np.linalg.norm(e0_changes)
    corr_e0 = float(np.sum(a_changes * e0_changes) / denom) if denom else 0.0
    assert abs(corr_e0) < 0.25, f"Uncorrelated Eve had unexpected high temporal correlation: {corr_e0:.3f}"


if __name__ == "__main__":
    test_exact_static_reciprocity()
    test_doppler_temporal_evolution()
    test_null_masking()
    test_session_shapes_and_eve_isolation()
    print("All Wi-Fi IoT channel simulator verification tests passed!")
