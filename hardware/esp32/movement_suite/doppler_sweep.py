"""Doppler Movement Sensitivity Sweep: Evaluates channel entropy and key agreement across motion speeds."""

from pathlib import Path
import numpy as np
import torch

from wifi_channel_sim import SimConfig, simulate_paired_iot_session
from wifi_model import build_wifi_resonance_model
import securelink

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_PATH = ROOT / "output/wifi_training/candidate_a/best_weights.pt"


def run_doppler_sweep(doppler_list=None, num_sessions=20, snr_db=25.0, seed=42):
    """
    Evaluates SecureLink across a spectrum of Doppler movement speeds:
    0.0 Hz (Stationary) -> 50.0 Hz (High speed).
    """
    if doppler_list is None:
        doppler_list = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load Candidate A neural denoiser if available
    model = None
    if WEIGHTS_PATH.exists():
        try:
            model = build_wifi_resonance_model(subcarriers=64, residual=True, safety_factor=2.0)
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
            model.eval()
        except Exception:
            model = None

    auth_secret = b"\x5a" * 32
    transcript = b"doppler-movement-test-v1"

    results = []

    print("\n" + "=" * 78)
    print("TEST SUITE 1: PHYSICAL DOPPLER VELOCITY & MOVEMENT SPECTRUM SWEEP")
    print("=" * 78)
    print(f"Sessions per setting: {num_sessions} | Operating SNR: {snr_db} dB | Rounds: 4 | Subcarriers: 64")
    print("-" * 78)
    print(f"{'Doppler (Hz)':<14} {'Motion Regime':<16} {'Variance (V/P)':<16} {'Disagreement':<14} {'Success %':<12} {'Status'}")
    print("-" * 78)

    for doppler in doppler_list:
        if doppler == 0.0:
            regime = "Stationary (0 m/s)"
        elif doppler <= 2.0:
            regime = "Micro-motion"
        elif doppler <= 12.0:
            regime = "Pedestrian walk"
        elif doppler <= 25.0:
            regime = "Running / brisk"
        else:
            regime = "Fast machinery"

        config = SimConfig(
            subcarriers=64,
            max_doppler_hz=doppler,
            num_paths=12 if doppler > 0 else 6,
        )

        handshake_success = 0
        fallback_count = 0
        raw_disagreements = []
        denoised_disagreements = []
        variances = []

        for sess_idx in range(num_sessions):
            nonce = (sess_idx + 1).to_bytes(16, "big")
            rng = np.random.default_rng(seed + sess_idx + int(doppler * 100))

            # Simulate paired IoT session (4 rounds x 8 packets = 32 frames)
            alice_frames, bob_frames, _ = simulate_paired_iot_session(
                config=config,
                packets_per_round=8,
                num_rounds=4,
                reciprocity_delay_sec=0.002,
                packet_interval_sec=0.010,
                round_interval_sec=0.100,
                snr_db=snr_db,
                eve_correlations=(0.0,),
                rng=rng,
            )

            # Measure relative temporal variance
            alice_mag = np.linalg.norm(alice_frames, axis=-1)
            dynamic = alice_mag - alice_mag.mean(axis=0, keepdims=True)
            var = float(np.mean(dynamic ** 2))
            pwr = float(np.mean(alice_mag ** 2))
            rel_var = var / max(pwr, 1e-7)
            variances.append(rel_var)

            # Physical Key Agreement evaluation
            # SecureLink's motion entropy gate enforces that stationary channels (V/P < 0.01)
            # gracefully fall back to authenticated crypto, preventing key extraction on static channels.
            try:
                offer, alice_sess = securelink.create_alice_offer(
                    alice_frames, auth_secret, nonce, transcript,
                    min_relative_variance=0.01, guard_ratio=0.15, min_bits=256
                )
                seen = set()
                reply, bob_sess = securelink.accept_alice_offer(
                    bob_frames, offer, auth_secret, nonce, transcript, seen,
                    min_relative_variance=0.01, guard_ratio=0.15, min_bits=256
                )
                final_key = securelink.finalize_alice_session(alice_sess, reply)
                assert final_key == bob_sess.key

                handshake_success += 1
                raw_disagreements.append(bob_sess.correction_rate)
            except ValueError:
                fallback_count += 1

            # Denoised CSI evaluation with WiFiResonanceModel
            if model is not None:
                with torch.no_grad():
                    orig_shape = alice_frames.shape
                    a_flat = alice_frames.reshape(-1, 64, 2)
                    b_flat = bob_frames.reshape(-1, 64, 2)
                    a_in = torch.from_numpy(a_flat).permute(0, 2, 1).float()
                    b_in = torch.from_numpy(b_flat).permute(0, 2, 1).float()
                    snr_t = torch.full((len(a_in),), float(snr_db))
                    a_den = model(a_in, snr_t).permute(0, 2, 1).numpy().reshape(orig_shape)
                    b_den = model(b_in, snr_t).permute(0, 2, 1).numpy().reshape(orig_shape)

                    # Subcarrier disagreement on denoised features
                    a_feat = np.linalg.norm(a_den, axis=-1).mean(axis=(0, 1))
                    b_feat = np.linalg.norm(b_den, axis=-1).mean(axis=(0, 1))
                    a_bits = a_feat >= np.median(a_feat)
                    b_bits = b_feat >= np.median(b_feat)
                    denoised_disagreements.append(float(np.mean(a_bits != b_bits)))

        avg_var = float(np.mean(variances))
        success_rate = (handshake_success / num_sessions) * 100.0
        avg_raw_dis = float(np.mean(raw_disagreements)) if raw_disagreements else 0.0

        if doppler == 0.0:
            status = "PASSED (Clean Fallback)" if fallback_count > 0 else "ACTIVE"
        elif success_rate >= 80.0:
            status = "PASSED (Active Key)"
        else:
            status = "COHERENCE LIMIT"

        print(f"{doppler:<14.1f} {regime:<16} {avg_var:<16.6f} {avg_raw_dis * 100:<13.2f}% {success_rate:<11.1f}% {status}")

        results.append({
            "doppler_hz": doppler,
            "regime": regime,
            "relative_variance": avg_var,
            "handshake_success_rate": success_rate,
            "fallback_count": fallback_count,
            "raw_disagreement": avg_raw_dis,
            "denoised_disagreement": float(np.mean(denoised_disagreements)) if denoised_disagreements else 0.0,
            "status": status,
        })

    print("-" * 78)
    return results
