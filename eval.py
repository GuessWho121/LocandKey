"""Evaluate the multi-scenario denoiser with a deterministic SNR sweep."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import build_resonance_model
from preprocess import DATA_ROOT, SPLITS_DIR, add_awgn, load_split_files, normalize_batch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "weights", "locandkey_multiscenario_best.weights.h5")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "visualizations", "multiscenario")
DEFAULT_SNRS = [-10, -5, 0, 5, 10, 15, 20, 25, 30]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LocandKey across scenarios and SNRs.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--split-dir", default=SPLITS_DIR)
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--snr-db", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--max-samples-per-scenario", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def empty_totals():
    return {
        "signal": 0.0,
        "model_error": 0.0,
        "noisy_error": 0.0,
        "model_absolute_error": 0.0,
        "noisy_absolute_error": 0.0,
        "model_power": 0.0,
        "noisy_power": 0.0,
        "model_cross_real": 0.0,
        "model_cross_imag": 0.0,
        "noisy_cross_real": 0.0,
        "noisy_cross_imag": 0.0,
        "elements": 0,
        "samples": 0,
    }


def accumulate(totals, clean, noisy, prediction):
    true_complex = clean[..., 0] + 1j * clean[..., 1]
    noisy_complex = noisy[..., 0] + 1j * noisy[..., 1]
    model_complex = prediction[..., 0] + 1j * prediction[..., 1]
    model_error = true_complex - model_complex
    noisy_error = true_complex - noisy_complex
    model_cross = np.vdot(true_complex, model_complex)
    noisy_cross = np.vdot(true_complex, noisy_complex)
    totals["signal"] += float(np.sum(np.abs(true_complex) ** 2, dtype=np.float64))
    totals["model_error"] += float(np.sum(np.abs(model_error) ** 2, dtype=np.float64))
    totals["noisy_error"] += float(np.sum(np.abs(noisy_error) ** 2, dtype=np.float64))
    totals["model_absolute_error"] += float(np.sum(np.abs(model_error), dtype=np.float64))
    totals["noisy_absolute_error"] += float(np.sum(np.abs(noisy_error), dtype=np.float64))
    totals["model_power"] += float(np.sum(np.abs(model_complex) ** 2, dtype=np.float64))
    totals["noisy_power"] += float(np.sum(np.abs(noisy_complex) ** 2, dtype=np.float64))
    totals["model_cross_real"] += float(model_cross.real)
    totals["model_cross_imag"] += float(model_cross.imag)
    totals["noisy_cross_real"] += float(noisy_cross.real)
    totals["noisy_cross_imag"] += float(noisy_cross.imag)
    totals["elements"] += true_complex.size
    totals["samples"] += len(clean)


def merge_totals(target, source):
    for key, value in source.items():
        target[key] += value


def final_metrics(totals, scenario, snr_db):
    model_cross = complex(totals["model_cross_real"], totals["model_cross_imag"])
    noisy_cross = complex(totals["noisy_cross_real"], totals["noisy_cross_imag"])
    model_nmse = 10.0 * np.log10(max(totals["model_error"] / totals["signal"], 1e-30))
    noisy_nmse = 10.0 * np.log10(max(totals["noisy_error"] / totals["signal"], 1e-30))
    return {
        "scenario": scenario,
        "snr_db": snr_db,
        "samples": totals["samples"],
        "noisy_nmse_db": float(noisy_nmse),
        "model_nmse_db": float(model_nmse),
        "nmse_gain_db": float(noisy_nmse - model_nmse),
        "noisy_mae": totals["noisy_absolute_error"] / totals["elements"],
        "model_mae": totals["model_absolute_error"] / totals["elements"],
        "noisy_correlation": float(abs(noisy_cross) / max(np.sqrt(totals["signal"] * totals["noisy_power"]), 1e-30)),
        "model_correlation": float(abs(model_cross) / max(np.sqrt(totals["signal"] * totals["model_power"]), 1e-30)),
    }


def plot_results(rows, output_dir):
    overall = sorted((row for row in rows if row["scenario"] == "overall"), key=lambda row: row["snr_db"])
    plt.figure(figsize=(9, 5))
    plt.plot([row["snr_db"] for row in overall], [row["noisy_nmse_db"] for row in overall], "o-", label="Noisy CSI")
    plt.plot([row["snr_db"] for row in overall], [row["model_nmse_db"] for row in overall], "s-", label="AI denoised")
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("NMSE (dB, lower is better)")
    plt.title("Multi-scenario CSI denoising")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nmse_vs_snr.png"), dpi=150)
    plt.close()

    scenarios = sorted({row["scenario"] for row in rows if row["scenario"] != "overall"})
    selected_snrs = [snr for snr in (0.0, 10.0) if any(row["snr_db"] == snr for row in rows)]
    if selected_snrs:
        x = np.arange(len(scenarios))
        width = 0.35
        plt.figure(figsize=(11, 5))
        for position, snr in enumerate(selected_snrs):
            gains = [next(row["nmse_gain_db"] for row in rows if row["scenario"] == name and row["snr_db"] == snr)
                     for name in scenarios]
            plt.bar(x + (position - (len(selected_snrs) - 1) / 2) * width, gains, width, label=f"{snr:g} dB")
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(x, scenarios, rotation=20, ha="right")
        plt.ylabel("NMSE improvement (dB)")
        plt.title("Denoising gain by scenario")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gain_by_scenario.png"), dpi=150)
        plt.close()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not os.path.isfile(args.weights_path):
        raise FileNotFoundError(f"Weights not found: {args.weights_path}")

    tf.keras.utils.set_random_seed(args.seed)
    model = build_resonance_model((128, 256, 2))
    model.load_weights(args.weights_path)
    summary, indices_file, channels = load_split_files(args.data_root, args.split_dir)
    rows = []

    for snr_position, snr_db in enumerate(args.snr_db):
        overall = empty_totals()
        for scenario_position, scenario in enumerate(summary["scenarios"]):
            indices = indices_file[f"test__{scenario}"]
            if args.max_samples_per_scenario:
                indices = indices[:args.max_samples_per_scenario]
            totals = empty_totals()
            for batch_number, start in enumerate(range(0, len(indices), args.batch_size)):
                selected = indices[start:start + args.batch_size]
                clean = normalize_batch(channels[scenario][selected])
                rng = np.random.default_rng(args.seed + snr_position * 100_003 + scenario_position * 1_009 + batch_number)
                noisy = add_awgn(clean, np.full(len(clean), snr_db), rng)
                prediction = np.asarray(model.predict_on_batch(noisy), dtype=np.float32)
                if not np.isfinite(prediction).all():
                    raise ValueError(f"Non-finite prediction for {scenario} at {snr_db:g} dB")
                accumulate(totals, clean, noisy, prediction)
            merge_totals(overall, totals)
            result = final_metrics(totals, scenario, snr_db)
            rows.append(result)
            print(f"{snr_db:>5g} dB {scenario:24s} model={result['model_nmse_db']:7.2f} dB gain={result['nmse_gain_db']:7.2f} dB")
        rows.append(final_metrics(overall, "overall", snr_db))
    indices_file.close()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["scenario"], row["snr_db"]): row for row in rows}
    required = [snr for snr in (0.0, 10.0) if ("overall", snr) in lookup]
    scenarios = list(summary["scenarios"])
    acceptance = {
        "overall_gain_at_least_2_db": bool(required) and all(lookup[("overall", snr)]["nmse_gain_db"] >= 2 for snr in required),
        "positive_gain_every_scenario": bool(required) and all(
            lookup[(scenario, snr)]["nmse_gain_db"] > 0 for scenario in scenarios for snr in required
        ),
    }
    report = {"weights": os.path.abspath(args.weights_path), "snr_db": args.snr_db, "acceptance": acceptance, "results": rows}
    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    plot_results(rows, args.output_dir)
    print(f"Metrics: {csv_path}")
    print(f"Acceptance: {acceptance}")


if __name__ == "__main__":
    main()
