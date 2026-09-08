"""Evaluate the multi-scenario denoiser with a deterministic SNR sweep."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_resonance_model
from preprocess import DATA_ROOT, SPLITS_DIR, add_awgn, load_split_files, normalize_batch, split_fingerprint


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "weights", "locandkey_multiscenario_best.pt")
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
    parser.add_argument("--baseline-weights", help="Re-evaluate the original model on identical samples and noise.")
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
    if totals["samples"] == 0 or totals["signal"] <= 0:
        raise ValueError(f"No nonzero test samples for {scenario}")
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


def evaluate_indices(model, baseline, channel, indices, args, device, snr_position, scenario_position, snr_db):
    totals, baseline_totals = empty_totals(), empty_totals()
    if args.max_samples_per_scenario:
        indices = indices[:args.max_samples_per_scenario]
    for batch_number, start in enumerate(range(0, len(indices), args.batch_size)):
        clean = normalize_batch(channel[indices[start:start + args.batch_size]])
        rng = np.random.default_rng(args.seed + snr_position * 100_003 + scenario_position * 1_009 + batch_number)
        noisy = add_awgn(clean, np.full(len(clean), snr_db), rng)
        with torch.inference_mode():
            inputs = torch.from_numpy(noisy).permute(0, 3, 1, 2).to(device)
            for network, target in ((model, totals), (baseline, baseline_totals)):
                if network is not None:
                    prediction = network(inputs).permute(0, 2, 3, 1).cpu().numpy()
                    if not np.isfinite(prediction).all():
                        raise ValueError(f"Non-finite prediction at {snr_db:g} dB")
                    accumulate(target, clean, noisy, prediction)
    return totals, baseline_totals


def holdout_decisions(unseen, comparisons):
    required = {0.0, 10.0}
    at_targets = {row["snr_db"]: row for row in unseen if row["snr_db"] in required}
    return {
        "unseen_gain_at_least_2_db": (all(row["nmse_gain_db"] >= 2 for row in at_targets.values())
                                        if set(at_targets) == required else None),
        "unseen_worsened_snrs": [row["snr_db"] for row in unseen if row["nmse_gain_db"] < 0],
        "seen_degradation_over_1_db": [row for row in comparisons if row["scope"] == "seen"
                                      and row["snr_db"] in required and row["degradation_db"] > 1],
        "baseline_comparison_available": bool(comparisons),
        "interpretation": "Review validation convergence before attributing errors to generalization; do not tune against the held-out scenario.",
    }


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not os.path.isfile(args.weights_path):
        raise FileNotFoundError(f"Weights not found: {args.weights_path}")

    if args.max_samples_per_scenario is not None and args.max_samples_per_scenario <= 0:
        raise SystemExit("--max-samples-per-scenario must be positive")
    if args.weights_path.endswith(".h5"):
        raise SystemExit("TensorFlow .h5 weights are incompatible; train a PyTorch .pt checkpoint first")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resonance_model((128, 256, 2)).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Device: {device}")
    summary, indices_file, channels = load_split_files(args.data_root, args.split_dir)
    holdout = summary.get("holdout_scenario")
    if holdout and os.path.normcase(os.path.realpath(args.output_dir)) == os.path.normcase(os.path.realpath(DEFAULT_OUTPUT)):
        raise SystemExit("Use a separate --output-dir for holdout evaluation")
    baseline = None
    if args.baseline_weights:
        baseline = build_resonance_model((128, 256, 2)).to(device)
        baseline.load_state_dict(torch.load(args.baseline_weights, map_location=device, weights_only=True))
        baseline.eval()
    rows = []
    group_rows, original_rows, comparisons = [], [], []

    for snr_position, snr_db in enumerate(args.snr_db):
        overall = empty_totals()
        seen = empty_totals()
        for scenario_position, scenario in enumerate(summary["scenarios"]):
            indices = indices_file[f"test__{scenario}"]
            totals, baseline_totals = evaluate_indices(model, baseline if scenario != holdout else None,
                channels[scenario], indices, args, device, snr_position, scenario_position, snr_db)
            merge_totals(overall, totals)
            if holdout and scenario != holdout:
                merge_totals(seen, totals)
            result = final_metrics(totals, scenario, snr_db)
            rows.append(result)
            if holdout:
                comparison_result = result
                scope = "seen"
                if scenario == holdout:
                    group_rows.append({**result, "scenario": "unseen"})
                    original, baseline_totals = evaluate_indices(model, baseline, channels[scenario],
                        indices_file[summary["original_test_key"]], args, device, snr_position, scenario_position, snr_db)
                    comparison_result = final_metrics(original, scenario, snr_db)
                    original_rows.append(comparison_result)
                    scope = "original_holdout_test"
                if baseline is not None:
                    old = final_metrics(baseline_totals, scenario, snr_db)
                    comparisons.append({"scenario": scenario, "scope": scope, "snr_db": snr_db,
                        "samples": comparison_result["samples"], "baseline_nmse_db": old["model_nmse_db"],
                        "model_nmse_db": comparison_result["model_nmse_db"],
                        "degradation_db": comparison_result["model_nmse_db"] - old["model_nmse_db"]})
            print(f"{snr_db:>5g} dB {scenario:24s} model={result['model_nmse_db']:7.2f} dB gain={result['nmse_gain_db']:7.2f} dB")
        rows.append(final_metrics(overall, "overall", snr_db))
        if holdout:
            group_rows.append(final_metrics(seen, "seen", snr_db))
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
        "overall_gain_at_least_2_db": len(required) == 2 and all(lookup[("overall", snr)]["nmse_gain_db"] >= 2 for snr in required),
        "positive_gain_every_scenario": len(required) == 2 and all(
            lookup[(scenario, snr)]["nmse_gain_db"] > 0 for scenario in scenarios for snr in required
        ),
    }
    report = {"weights": os.path.abspath(args.weights_path), "snr_db": args.snr_db, "acceptance": acceptance, "results": rows}
    report.update(split_sha256=split_fingerprint(args.split_dir), seed=args.seed, batch_size=args.batch_size,
                  max_samples_per_scenario=args.max_samples_per_scenario)
    if holdout:
        unseen = [row for row in group_rows if row["scenario"] == "unseen"]
        report.update(holdout_scenario=holdout, groups=group_rows, original_holdout_test=original_rows,
                      baseline_weights=args.baseline_weights, comparisons=comparisons,
                      holdout_decisions=holdout_decisions(unseen, comparisons))
        for name, values in (("group_metrics", group_rows), ("original_holdout_test", original_rows), ("baseline_comparison", comparisons)):
            if values:
                with open(os.path.join(args.output_dir, name + ".csv"), "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)
        plt.figure(figsize=(9, 5))
        ordered = sorted(unseen, key=lambda row: row["snr_db"])
        for field, label in (("noisy_nmse_db", "Noisy CSI"), ("model_nmse_db", "AI denoised")):
            plt.plot([row["snr_db"] for row in ordered], [row[field] for row in ordered], "o-", label=label)
        plt.xlabel("Input SNR (dB)")
        plt.ylabel("NMSE (dB, lower is better)")
        plt.title(f"Unseen environment: {holdout}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "holdout_nmse_vs_snr.png"), dpi=150)
        plt.close()
        print(f"Holdout decisions: {report['holdout_decisions']}")
    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    plot_results(rows, args.output_dir)
    print(f"Metrics: {csv_path}")
    print(f"Acceptance: {acceptance}")


if __name__ == "__main__":
    main()
