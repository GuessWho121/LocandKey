"""Deterministic Alice/Bob/Eve operating-envelope evaluation for SecureLink."""

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np


def normalize_batch(batch):
    batch = np.asarray(batch, dtype=np.float32)
    norm = np.sqrt(np.sum(batch * batch, axis=(1, 2, 3), keepdims=True, dtype=np.float64)).astype(np.float32)
    return batch / np.maximum(norm, np.float32(1e-10))


from securelink import (BCH_ECC_BITS, _local_observation, accept_alice_offer, create_alice_offer,
                        finalize_alice_session)


DEFAULT_SNRS = tuple(range(-10, 31, 5))
DEFAULT_EVE_CORRELATIONS = (0.0, 0.25, 0.5, 0.75, 0.9)
SECRET = hashlib.sha256(b"securelink-evaluation-authenticated-secret").digest()


def add_awgn(channel, snr_db, rng):
    power = float(np.mean(channel ** 2))
    scale = math.sqrt(power / (10 ** (snr_db / 10.0)))
    return channel + rng.normal(scale=scale, size=channel.shape).astype(np.float32)


def make_session(rng, snr_db, frames, rows, columns, eve_correlations):
    shape = (rows, columns, 2)
    base = rng.normal(size=shape).astype(np.float32)
    motion = rng.normal(size=shape).astype(np.float32)
    time_axis = np.sin(np.linspace(0, 4 * np.pi, frames, dtype=np.float32))[:, None, None, None]
    shared = normalize_batch(base + 0.2 * time_axis * motion)
    alice = add_awgn(shared, snr_db, rng)
    bob = add_awgn(shared, snr_db, rng)
    eves = []
    for correlation in eve_correlations:
        independent = normalize_batch(
            rng.normal(size=(frames, *shape)).astype(np.float32)
            + 0.2 * time_axis * rng.normal(size=shape).astype(np.float32))
        eve_clean = correlation * shared + math.sqrt(1.0 - correlation ** 2) * independent
        eves.append(add_awgn(eve_clean, snr_db, rng))
    return alice, bob, eves


def make_physical_session(rng, snr_db, frames, subcarriers, eve_correlations,
                          rounds=4, doppler_hz=10.0, delay_ms=2.0, receiver_mismatch=False):
    from wifi_channel_sim import SimConfig, ReceiverImpairment, simulate_paired_iot_session
    config = SimConfig(subcarriers=subcarriers, max_doppler_hz=doppler_hz)
    imp_b = (ReceiverImpairment(gain=float(rng.uniform(0.80, 1.15)),
                                slope=float(rng.uniform(-0.02, 0.02)))
             if receiver_mismatch else None)
    alice_iq, bob_iq, eves_iq = simulate_paired_iot_session(
        config,
        packets_per_round=frames,
        num_rounds=rounds,
        reciprocity_delay_sec=delay_ms * 1e-3,
        snr_db=snr_db,
        alice_impairment=None,
        bob_impairment=imp_b,
        eve_correlations=tuple(eve_correlations),
        rng=rng,
    )
    return alice_iq, bob_iq, eves_iq


def load_denoiser(weights_path, direct_output, safety_factor, wifi_iot=False):
    if not weights_path:
        return None, None, None
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wifi_iot:
        from wifi_model import build_wifi_resonance_model
        model = build_wifi_resonance_model(residual=not direct_output, safety_factor=safety_factor).to(device)
    else:
        raise ValueError("Legacy 2D models have been superseded; run with --wifi-iot")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model, device, torch


def denoise_channels(channels, snr_db, model, device, torch, wifi_iot=False):
    lengths = [len(channel) for channel in channels]
    combined = np.concatenate(channels)
    started = time.perf_counter()
    with torch.inference_mode():
        if wifi_iot:
            orig_shape = combined.shape
            flat_combined = combined.reshape(-1, orig_shape[-2], orig_shape[-1])
            inputs = torch.from_numpy(flat_combined).permute(0, 2, 1).to(device)
            snr = torch.full((len(inputs),), float(snr_db), device=device)
            output = model(inputs, snr).permute(0, 2, 1).cpu().numpy().reshape(orig_shape)
        else:
            inputs = torch.from_numpy(combined).permute(0, 3, 1, 2).to(device)
            snr = torch.full((len(inputs),), float(snr_db), device=device)
            output = model(inputs, snr).permute(0, 2, 3, 1).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Denoiser produced non-finite CSI")
    boundaries = np.cumsum(lengths)[:-1]
    return np.split(output, boundaries), (time.perf_counter() - started) * 1000


def offer_metrics(alice, bob, offer):
    alice_feature, _ = _local_observation(alice)
    bob_feature, _ = _local_observation(bob)
    reliability = np.unpackbits(np.frombuffer(offer.reliability_mask, dtype=np.uint8),
                                count=offer.feature_count).astype(bool)
    alice_bits = alice_feature[reliability] >= 0
    bob_bits = bob_feature[reliability] >= 0
    raw_bit_errors = int(np.count_nonzero(alice_bits != bob_bits))
    disagreement = raw_bit_errors / len(alice_bits)
    ones = float(alice_bits.mean())
    entropy_per_bit = -math.log2(max(ones, 1.0 - ones))
    leakage = BCH_ECC_BITS
    return disagreement, raw_bit_errors, max(0.0, len(alice_bits) * entropy_per_bit - leakage)


def run_handshake(alice, bob, nonce, transcript):
    started = time.perf_counter()
    try:
        offer, alice_session = create_alice_offer(alice, SECRET, nonce, transcript)
    except ValueError as error:
        return {
            "success": 0, "fallback_reason": str(error),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "denoising_ms": 0.0,
            "silent_mismatch": 0, "tamper_accepted": 0, "replay_accepted": 0,
            "raw_disagreement_rate": None, "raw_bit_errors": None, "correction_rate": None,
            "retained_bits": 0, "helper_leakage_bits": 0,
            "estimated_min_entropy_bits": 0.0,
        }, None, None

    disagreement, raw_bit_errors, min_entropy = offer_metrics(alice, bob, offer)
    retained_bits = int(np.count_nonzero(np.unpackbits(
        np.frombuffer(offer.reliability_mask, dtype=np.uint8), count=offer.feature_count)))
    seen_nonces = set()
    try:
        reply, bob_session = accept_alice_offer(
            bob, offer, SECRET, nonce, transcript, seen_nonces)
    except ValueError as error:
        return {
            "success": 0, "fallback_reason": str(error),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "denoising_ms": 0.0,
            "silent_mismatch": 0, "tamper_accepted": 0, "replay_accepted": 0,
            "raw_disagreement_rate": disagreement, "raw_bit_errors": raw_bit_errors,
            "correction_rate": None, "retained_bits": retained_bits,
            "helper_leakage_bits": BCH_ECC_BITS,
            "estimated_min_entropy_bits": min_entropy,
        }, None, None

    success, fallback_reason, silent_mismatch = 1, "", 0
    try:
        alice_key = finalize_alice_session(alice_session, reply)
        silent_mismatch = int(alice_key != bob_session.key)
    except ValueError as error:
        success, fallback_reason = 0, str(error)
    latency_ms = (time.perf_counter() - started) * 1000

    replay_accepted = 0
    try:
        accept_alice_offer(bob, offer, SECRET, nonce, transcript, seen_nonces)
        replay_accepted = 1
    except ValueError:
        pass
    tamper_accepted = 0
    tampered = replace(offer, helper_data=bytes([offer.helper_data[0] ^ 1]) + offer.helper_data[1:])
    try:
        accept_alice_offer(bob, tampered, SECRET, nonce, transcript, set())
        tamper_accepted = 1
    except ValueError:
        pass
    return {
        "success": success, "fallback_reason": fallback_reason,
        "latency_ms": latency_ms, "denoising_ms": 0.0,
        "silent_mismatch": silent_mismatch, "tamper_accepted": tamper_accepted,
        "replay_accepted": replay_accepted, "raw_disagreement_rate": disagreement,
        "raw_bit_errors": raw_bit_errors,
        "correction_rate": bob_session.correction_rate,
        "retained_bits": retained_bits,
        "helper_leakage_bits": bob_session.helper_leakage_bits,
        "estimated_min_entropy_bits": min_entropy,
    }, (offer if success else None), (alice_session if success else None)


def eve_confirms(eve, offer, alice_session, nonce, transcript):
    if offer is None:
        return 0
    try:
        reply, eve_session = accept_alice_offer(eve, offer, SECRET, nonce, transcript, set())
        return int(finalize_alice_session(alice_session, reply) == eve_session.key)
    except ValueError:
        return 0


def aggregate(session_rows):
    grouped = {}
    for row in session_rows:
        grouped.setdefault((row["mode"], row["snr_db"], row["eve_correlation"]), []).append(row)
    metrics = []
    for (mode, snr_db, eve_correlation), rows in sorted(grouped.items()):
        successes = [row for row in rows if row["success"]]
        measured = [row for row in rows if row["raw_disagreement_rate"] is not None]
        p95_latency = float(np.percentile([row["latency_ms"] for row in rows], 95))
        def mean(key):
            values = [row[key] for row in rows if row[key] is not None]
            return float(np.mean(values)) if values else 0.0
        reasons = Counter(row["fallback_reason"] for row in rows if row["fallback_reason"])
        metrics.append({
            "mode": mode, "snr_db": snr_db, "eve_correlation": eve_correlation,
            "sessions": len(rows), "handshake_success_rate": len(successes) / len(rows),
            "fallback_rate": 1.0 - len(successes) / len(rows),
            "raw_disagreement_rate": mean("raw_disagreement_rate"),
            "p95_raw_bit_errors": (float(np.percentile(
                [row["raw_bit_errors"] for row in measured], 95)) if measured else 0.0),
            "correction_rate": mean("correction_rate"),
            "mean_retained_bits": mean("retained_bits"),
            "mean_helper_leakage_bits": mean("helper_leakage_bits"),
            "mean_estimated_min_entropy_bits": mean("estimated_min_entropy_bits"),
            "usable_key_bits_per_session": sum(
                max(0.0, row["estimated_min_entropy_bits"]) for row in successes) / len(rows),
            "p95_latency_ms": p95_latency,
            "p95_denoising_ms": float(np.percentile([row["denoising_ms"] for row in rows], 95)),
            "silent_key_mismatches": sum(row["silent_mismatch"] for row in rows),
            "tamper_accepts": sum(row["tamper_accepted"] for row in rows),
            "replay_accepts": sum(row["replay_accepted"] for row in rows),
            "eve_confirmed_keys": sum(row["eve_confirmed"] for row in rows),
            "eve_confirmation_rate": sum(row["eve_confirmed"] for row in rows) / len(rows),
            "fallback_reasons": json.dumps(reasons, sort_keys=True),
        })
    return metrics


def acceptance(metrics, usable_snr_min):
    primary = [row for row in metrics if row["mode"] == "raw" and row["eve_correlation"] == 0.0]
    usable = [row for row in primary if row["snr_db"] >= usable_snr_min]
    all_rows = list(metrics)
    checks = {
        "zero_silent_key_mismatches": sum(row["silent_key_mismatches"] for row in all_rows) == 0,
        "zero_tamper_accepts": sum(row["tamper_accepts"] for row in all_rows) == 0,
        "zero_replay_accepts": sum(row["replay_accepts"] for row in all_rows) == 0,
        "usable_region_success_at_least_95_percent": bool(usable) and all(
            row["handshake_success_rate"] >= 0.95 for row in usable),
        "zero_confirmed_eve_keys": sum(row["eve_confirmed_keys"] for row in all_rows) == 0,
    }
    checks["pass"] = all(checks.values())
    return checks


def select_mode(metrics):
    modes = {row["mode"] for row in metrics}
    if "denoised" not in modes:
        return "raw"
    lookup = {(row["mode"], row["snr_db"], row["eve_correlation"]): row for row in metrics}
    raw_keys = [(snr, rho) for mode, snr, rho in lookup if mode == "raw"]
    safe = all(
        lookup[("denoised", snr, rho)]["handshake_success_rate"]
        >= lookup[("raw", snr, rho)]["handshake_success_rate"]
        and lookup[("denoised", snr, rho)]["eve_confirmation_rate"]
        <= lookup[("raw", snr, rho)]["eve_confirmation_rate"]
        for snr, rho in raw_keys)
    improved = any(
        lookup[("denoised", snr, rho)]["handshake_success_rate"]
        > lookup[("raw", snr, rho)]["handshake_success_rate"] for snr, rho in raw_keys)
    return "denoised" if safe and improved else "raw"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(metrics, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted({row["mode"] for row in metrics})
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for mode in modes:
        rows = sorted((row for row in metrics if row["mode"] == mode
                       and row["eve_correlation"] == 0.0), key=lambda row: row["snr_db"])
        axes[0, 0].plot([row["snr_db"] for row in rows],
                        [row["handshake_success_rate"] for row in rows], "o-", label=mode)
        axes[0, 1].plot([row["snr_db"] for row in rows],
                        [row["raw_disagreement_rate"] for row in rows], "o-", label=mode)
        axes[1, 0].plot([row["snr_db"] for row in rows],
                        [row["p95_latency_ms"] for row in rows], "o-", label=mode)
        eve_rows = sorted((row for row in metrics if row["mode"] == mode and row["snr_db"] == 10),
                          key=lambda row: row["eve_correlation"])
        axes[1, 1].plot([row["eve_correlation"] for row in eve_rows],
                        [row["eve_confirmation_rate"] for row in eve_rows], "o-", label=mode)
    labels = (("Handshake success", "Rate"), ("Raw bit disagreement", "Rate"),
              ("Handshake latency", "p95 milliseconds"), ("Eve confirmation at 10 dB", "Rate"))
    for axis, (title, ylabel) in zip(axes.flat, labels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0, 0].set_xlabel("SNR (dB)")
    axes[0, 1].set_xlabel("SNR (dB)")
    axes[1, 0].set_xlabel("SNR (dB)")
    axes[1, 1].set_xlabel("Eve channel correlation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def evaluate(args):
    if args.trials <= 0 or args.frames < 4:
        raise ValueError("Trials and at least four frames are required")
    if not args.physical_sim and (args.rows <= 0 or args.columns <= 0):
        raise ValueError("Dimensions rows and columns are required for legacy evaluation")
    wifi_iot = bool(args.physical_sim)
    model, device, torch = load_denoiser(args.weights_path, args.direct_output, args.safety_factor, wifi_iot=wifi_iot)
    if model is not None and not wifi_iot and (args.rows, args.columns) != (128, 256):
        raise ValueError("Legacy denoised evaluation requires --rows 128 --columns 256")
    modes = ("raw", "denoised") if model is not None else ("raw",)
    session_rows = []
    for snr_position, snr_db in enumerate(args.snr_db):
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed + snr_position * 1_000_003 + trial)
            if args.physical_sim:
                channels = list(make_physical_session(
                    rng, snr_db, args.frames, args.subcarriers, args.eve_correlation,
                    doppler_hz=args.doppler_hz, delay_ms=args.reciprocity_delay_ms,
                    receiver_mismatch=args.receiver_mismatch,
                ))
            else:
                channels = list(make_session(rng, snr_db, args.frames, args.rows, args.columns,
                                             args.eve_correlation))
            raw_channels = [channels[0], channels[1], *channels[2]]
            mode_channels = {"raw": raw_channels}
            mode_denoising_ms = {"raw": 0.0}
            if model is not None:
                mode_channels["denoised"], total_denoising_ms = denoise_channels(
                    raw_channels, snr_db, model, device, torch, wifi_iot=wifi_iot)
                mode_denoising_ms["denoised"] = total_denoising_ms * 2 / len(raw_channels)
            for mode in modes:
                alice, bob, *eves = mode_channels[mode]
                nonce = snr_position.to_bytes(8, "big") + trial.to_bytes(8, "big")
                transcript = f"securelink-eval/{mode}/{snr_db:g}/{trial}".encode("ascii")
                result, offer, alice_session = run_handshake(alice, bob, nonce, transcript)
                result["denoising_ms"] = mode_denoising_ms[mode]
                result["latency_ms"] += mode_denoising_ms[mode]
                for correlation, eve in zip(args.eve_correlation, eves):
                    session_rows.append({
                        "mode": mode, "snr_db": snr_db, "eve_correlation": correlation,
                        "trial": trial, **result,
                        "eve_confirmed": eve_confirms(
                            eve, offer, alice_session, nonce, transcript),
                    })
        print(f"completed {args.trials} sessions at {snr_db:g} dB", flush=True)
    metrics = aggregate(session_rows)
    return session_rows, metrics, {
        "acceptance": acceptance(metrics, args.usable_snr_min),
        "selected_mode": select_mode(metrics),
        "eve_model": "worst-case Eve is given the authenticated secret to isolate channel similarity",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SecureLink across SNR and Eve correlation.")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--snr-db", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--eve-correlation", type=float, nargs="+", default=DEFAULT_EVE_CORRELATIONS)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--usable-snr-min", type=float, default=10.0)
    parser.add_argument("--weights-path")
    parser.add_argument("--direct-output", action="store_true")
    parser.add_argument("--safety-factor", type=float, default=2.0)
    parser.add_argument("--output-dir", default="output/securelink/evaluation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--physical-sim", action="store_true", help="Use physical paired Wi-Fi IoT channel simulator")
    parser.add_argument("--subcarriers", type=int, default=64, help="Subcarrier count for physical simulation")
    parser.add_argument("--doppler-hz", type=float, default=10.0, help="Max Doppler spread in Hz")
    parser.add_argument("--reciprocity-delay-ms", type=float, default=2.0, help="Reciprocity delay in ms")
    parser.add_argument("--receiver-mismatch", action="store_true", help="Include Bob receiver RF mismatch")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.trials, args.snr_db, args.eve_correlation = 2, (0.0, 10.0), (0.0, 0.9)
    if any(not 0.0 <= value < 1.0 for value in args.eve_correlation):
        raise SystemExit("--eve-correlation values must be in [0, 1)")
    if args.weights_path and not Path(args.weights_path).is_file():
        raise FileNotFoundError(args.weights_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    session_rows, metrics, decisions = evaluate(args)
    write_csv(output / "sessions.csv", session_rows)
    write_csv(output / "metrics.csv", metrics)
    plot_metrics(metrics, output / "protocol_metrics.png")
    report = {
        "config": vars(args),
        "provenance": {
            "seed": args.seed, "python": platform.python_version(), "numpy": np.__version__,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "securelink_sha256": hashlib.sha256(Path("securelink.py").read_bytes()).hexdigest(),
            "weights_sha256": (hashlib.sha256(Path(args.weights_path).read_bytes()).hexdigest()
                               if args.weights_path else None),
        },
        "decisions": decisions,
        "metrics": metrics,
    }
    (output / "evaluation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
