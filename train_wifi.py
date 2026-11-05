"""Training pipeline for Native Wi-Fi IoT CSI Denoiser (Candidates A, B, and C)."""

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wifi_model import build_wifi_resonance_model
from wifi_dataset import generate_wifi_iot_split, WIFI_SCENARIOS, WiFiScenarioSpec
from gate_aware_selector import evaluate_checkpoint_eligibility, select_best_checkpoint
from model import nmse_metric, nmse_db_metric, resonance_loss


def train_wifi_candidate(
    candidate_name: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    device_name: str = "auto",
    clean_fraction: float = 0.50,
    paired_consistency_weight: float = 0.0,
    output_dir: str = "output/wifi_training",
    seed: int = 42,
    smoke: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    out_path = Path(output_dir) / candidate_name
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Generation
    scenarios = [WiFiScenarioSpec(s.name, s.delay_spread_ns, s.rician_k_db, s.max_doppler_hz, 30 if smoke else s.sample_count)
                 for s in WIFI_SCENARIOS]
    train_ds, val_ds, test_ds = generate_wifi_iot_split(
        scenarios=scenarios,
        seed=seed,
        clean_fraction=clean_fraction,
    )

    # Tensor format: inputs are (N, 2, 64) [Real, Imag]
    x_train = torch.from_numpy(train_ds.samples).permute(0, 2, 1).float()
    y_train = torch.from_numpy(train_ds.clean_samples).permute(0, 2, 1).float()
    snr_train = torch.from_numpy(train_ds.snr_db).float()

    x_val = torch.from_numpy(val_ds.samples).permute(0, 2, 1).float()
    y_val = torch.from_numpy(val_ds.clean_samples).permute(0, 2, 1).float()
    snr_val = torch.from_numpy(val_ds.snr_db).float()

    train_loader = DataLoader(TensorDataset(x_train, y_train, snr_train), batch_size=batch_size, shuffle=True)

    # 2. Model & Optimizer
    model = build_wifi_resonance_model(subcarriers=64, residual=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = resonance_loss(lambda_mag=0.05)

    candidate_epochs = []
    print(f"Starting {candidate_name} ({epochs} epochs, train={len(train_ds)}, val={len(val_ds)}) on {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y, batch_snr in train_loader:
            batch_x, batch_y, batch_snr = batch_x.to(device), batch_y.to(device), batch_snr.to(device)
            optimizer.zero_grad()
            pred = model(batch_x, batch_snr)
            loss = loss_fn(batch_y, pred, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        scheduler.step()
        train_loss = total_loss / len(train_ds)

        # 3. Multi-SNR Validation
        model.eval()
        val_metrics = {}
        unique_snrs = np.unique(val_ds.snr_db)
        with torch.inference_mode():
            for snr_db in unique_snrs:
                mask = val_ds.snr_db == snr_db
                vx = x_val[mask].to(device)
                vy = y_val[mask].to(device)
                v_snr = snr_val[mask].to(device)
                v_pred = model(vx, v_snr)

                noisy_nmse = float(nmse_db_metric(vy, vx).item())
                model_nmse = float(nmse_db_metric(vy, v_pred).item())
                val_metrics[float(snr_db)] = {
                    "noisy_nmse_db": noisy_nmse,
                    "model_nmse_db": model_nmse,
                    "nmse_gain_db": noisy_nmse - model_nmse,
                }

        candidate_epochs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": np.mean([m["model_nmse_db"] for m in val_metrics.values()]),
            "metrics": val_metrics,
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        })

    # 4. Gate-Aware Checkpoint Selection
    status, best_candidate = select_best_checkpoint(candidate_epochs)
    report = {
        "candidate": candidate_name,
        "seed": seed,
        "epochs": epochs,
        "status": status,
        "best_epoch": best_candidate["epoch"] if best_candidate else None,
        "worst_gain_db": best_candidate["worst_gain"] if best_candidate else None,
        "history": [{k: v for k, v in e.items() if k != "state_dict"} for e in candidate_epochs],
    }

    if status == "selected" and best_candidate is not None:
        torch.save(best_candidate["state_dict"], out_path / "best_weights.pt")
        print(f"Checkpoint selected at Epoch {best_candidate['epoch']} (Worst Gain: {best_candidate['worst_gain']:.2f} dB)")
    else:
        print("Gate-aware check emitted: no_safe_candidate")

    (out_path / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="candidate_a", choices=["candidate_a", "candidate_b", "candidate_c"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    clean_frac = 0.50 if args.candidate != "candidate_a" else 1.0
    paired_weight = 0.05 if args.candidate == "candidate_c" else 0.0
    train_wifi_candidate(
        candidate_name=args.candidate,
        epochs=args.epochs,
        smoke=args.smoke,
        clean_fraction=clean_frac,
        paired_consistency_weight=paired_weight,
        device_name=args.device,
    )
