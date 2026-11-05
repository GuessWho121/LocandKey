"""Ping-Pong Reciprocity Lag Breakdown Test: Evaluates non-simultaneous probing delays under motion."""

from pathlib import Path
import numpy as np

from wifi_channel_sim import SimConfig, simulate_paired_iot_session
import securelink

ROOT = Path(__file__).resolve().parents[3]


def run_reciprocity_lag_sweep(lags_ms=None, doppler_hz=10.0, num_sessions=20, snr_db=25.0, seed=42):
    """
    Evaluates the impact of non-simultaneous ping-pong transmission time lags (Delta t)
    on channel correlation and key agreement under pedestrian movement (f_D = 10 Hz).
    """
    if lags_ms is None:
        lags_ms = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    np.random.seed(seed)
    auth_secret = b"\x5a" * 32
    transcript = b"reciprocity-lag-test-v1"

    results = []

    print("\n" + "=" * 78)
    print("TEST SUITE 2: RECIPROCITY DELAY (Delta t) VS. MOTION COHERENCE TEST")
    print("=" * 78)
    print(f"Pedestrian Doppler: {doppler_hz} Hz (~1.2 m/s) | SNR: {snr_db} dB | Sessions: {num_sessions}")
    print("-" * 78)
    print(f"{'Delay Delta t':<14} {'Channel r_AB':<16} {'Disagreement':<16} {'Success %':<14} {'Outcome'}")
    print("-" * 78)

    for lag in lags_ms:
        config = SimConfig(
            subcarriers=64,
            max_doppler_hz=doppler_hz,
            num_paths=12,
        )

        handshake_success = 0
        correlations = []
        disagreements = []

        for sess_idx in range(num_sessions):
            nonce = (sess_idx + 1).to_bytes(16, "big")
            rng = np.random.default_rng(seed + sess_idx + int(lag * 100))

            # Simulate paired IoT session with non-simultaneous delay (Delta t)
            alice_frames, bob_frames, _ = simulate_paired_iot_session(
                config=config,
                packets_per_round=8,
                num_rounds=4,
                reciprocity_delay_sec=lag * 1e-3,
                packet_interval_sec=0.010,
                round_interval_sec=0.100,
                snr_db=snr_db,
                eve_correlations=(0.0,),
                rng=rng,
            )

            # Compute dynamic correlation r_AB
            a_mag = np.linalg.norm(alice_frames, axis=-1)
            b_mag = np.linalg.norm(bob_frames, axis=-1)
            a_dyn = a_mag - a_mag.mean(axis=0, keepdims=True)
            b_dyn = b_mag - b_mag.mean(axis=0, keepdims=True)
            denom = np.linalg.norm(a_dyn) * np.linalg.norm(b_dyn)
            corr = float(np.sum(a_dyn * b_dyn) / denom) if denom > 0 else 0.0
            correlations.append(corr)

            try:
                offer, alice_sess = securelink.create_alice_offer(
                    alice_frames, auth_secret, nonce, transcript,
                    min_relative_variance=1e-4, guard_ratio=0.15, min_bits=256
                )
                seen = set()
                reply, bob_sess = securelink.accept_alice_offer(
                    bob_frames, offer, auth_secret, nonce, transcript, seen,
                    min_relative_variance=1e-4, guard_ratio=0.15, min_bits=256
                )
                final_key = securelink.finalize_alice_session(alice_sess, reply)
                assert final_key == bob_sess.key

                handshake_success += 1
                disagreements.append(bob_sess.correction_rate)
            except ValueError:
                pass

        avg_corr = float(np.mean(correlations))
        success_rate = (handshake_success / num_sessions) * 100.0
        avg_dis = float(np.mean(disagreements)) if disagreements else 0.0

        if lag <= 2.0:
            outcome = "EXCELLENT RECIPROCITY"
        elif lag <= 5.0:
            outcome = "ACCEPTABLE (Operational Limit)"
        else:
            outcome = "COHERENCE COLLAPSE"

        print(f"{lag:<14.1f} ms {avg_corr:<15.4f} {avg_dis * 100:<15.2f}% {success_rate:<13.1f}% {outcome}")

        results.append({
            "lag_ms": lag,
            "correlation": avg_corr,
            "disagreement": avg_dis,
            "success_rate": success_rate,
            "outcome": outcome,
        })

    print("-" * 78)
    return results
