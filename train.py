"""Train the LocandKey denoiser from memory-mapped multi-scenario CSI."""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import build_resonance_model, nmse_db_metric, nmse_metric, resonance_loss
from preprocess import DATA_ROOT, SPLITS_DIR, make_csi_sequence, split_fingerprint


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "weights", "locandkey_multiscenario_best.pt")
DEFAULT_LOGS = os.path.join(PROJECT_DIR, "logs", "multiscenario")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multi-scenario CSI denoiser.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--split-dir", default=SPLITS_DIR)
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS)
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--resume", nargs="?", const="auto", help="Resume a full checkpoint (default: best path + .last.pt).")
    restore.add_argument("--init-weights", help="Start a new training schedule from existing weights-only .pt file.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--snr-min-db", type=float, default=-10.0)
    parser.add_argument("--snr-max-db", type=float, default=30.0)
    parser.add_argument("--validation-snr-db", type=float, default=10.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run 64 train / 16 validation samples for one epoch.")
    return parser.parse_args()


def run_epoch(model, data, device, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    loss_fn = resonance_loss(lambda_mag=0.05)
    progress = tqdm(range(len(data)), desc="Train" if training else "Validation")
    for index in progress:
        noisy, clean = data[index]
        inputs = torch.from_numpy(noisy).permute(0, 3, 1, 2).to(device)
        targets = torch.from_numpy(clean).permute(0, 3, 1, 2).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=scaler is not None and scaler.is_enabled()):
                prediction = model(inputs)
            loss = loss_fn(targets, prediction)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {index}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
        values = (loss.detach().item(), nmse_metric(targets, prediction.detach()).item(),
                  nmse_db_metric(targets, prediction.detach()).item())
        totals += np.asarray(values) * len(clean)
        count += len(clean)
        progress.set_postfix(loss=f"{totals[0] / count:.4f}")
    return dict(zip(("loss", "nmse_metric", "nmse_db_metric"), totals / count))


def main():
    args = parse_args()
    checkpoint = None
    if args.resume:
        path = args.weights_path + ".last.pt" if args.resume == "auto" else args.resume
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise SystemExit("This is not a full checkpoint. Use --init-weights for old weights-only files.")
        # Restore the original schedule and data settings; output paths may move between laptops.
        for key in ("epochs", "batch_size", "patience", "learning_rate", "min_learning_rate",
                    "snr_min_db", "snr_max_db", "validation_snr_db", "max_train_samples",
                    "max_val_samples", "seed", "mixed_precision"):
            setattr(args, key, checkpoint["config"][key])
        if args.smoke:
            raise SystemExit("Resume smoke runs with explicit checkpoint/output paths, without --smoke")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("--batch-size and --epochs must be positive")
    if args.snr_min_db >= args.snr_max_db:
        raise SystemExit("--snr-min-db must be lower than --snr-max-db")
    if args.patience < 0 or not 0 <= args.min_learning_rate <= args.learning_rate or args.learning_rate <= 0:
        raise SystemExit("Invalid patience or learning-rate range")
    if args.weights_path.endswith(".h5"):
        raise SystemExit("Use a .pt weights path; TensorFlow .h5 weights are incompatible")

    if args.smoke:
        args.epochs = 1
        args.max_train_samples = 64
        args.max_val_samples = 16
        args.weights_path = os.path.join(PROJECT_DIR, "weights", "locandkey_smoke.pt")
        args.logs_dir = os.path.join(PROJECT_DIR, "logs", "smoke")

    random.seed(args.seed)
    with open(os.path.join(args.split_dir, "split_summary.json"), encoding="utf-8") as handle:
        split_summary = json.load(handle)
    fingerprint = split_fingerprint(args.split_dir)
    holdout = split_summary.get("holdout_scenario")
    if holdout:
        if args.init_weights:
            raise SystemExit("Holdout experiments must start fresh; --init-weights is not allowed")
        if os.path.abspath(args.weights_path) == os.path.abspath(DEFAULT_WEIGHTS) or os.path.abspath(args.logs_dir) == os.path.abspath(DEFAULT_LOGS):
            raise SystemExit("Use separate --weights-path and --logs-dir for the holdout experiment")
        if checkpoint is not None and checkpoint.get("split_sha256") != fingerprint:
            raise SystemExit("Resume checkpoint does not belong to these holdout splits")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise SystemExit("No PyTorch CUDA GPU is available; install the CUDA wheel documented in README.md")
    if args.mixed_precision and device.type != "cuda":
        raise SystemExit("--mixed-precision requires a CUDA GPU")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    train_data = make_csi_sequence(
        args.data_root, args.split_dir, "train", args.batch_size,
        seed=args.seed, shuffle=True, snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db, max_samples=args.max_train_samples,
    )
    validation_data = make_csi_sequence(
        args.data_root, args.split_dir, "validation", args.batch_size,
        seed=args.seed + 10_000, fixed_snr_db=args.validation_snr_db,
        max_samples=args.max_val_samples,
    )
    print(f"Training samples: {len(train_data.references):,}")
    print(f"Validation samples: {len(validation_data.references):,} at {args.validation_snr_db:g} dB")

    model = build_resonance_model((128, 256, 2)).to(device)
    total_steps = max(1, args.epochs * len(train_data))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    last_path = args.weights_path + ".last.pt"
    start_epoch, best_nmse, stale_epochs = 0, float("inf"), 0
    best_state = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["next_epoch"]
        best_nmse, stale_epochs = checkpoint["best_nmse"], checkpoint["stale_epochs"]
        best_state = checkpoint["best_model"]
        for _ in range(start_epoch):
            train_data.on_epoch_end()
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        state = checkpoint["numpy_rng"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        print(f"Resuming at epoch {start_epoch + 1}; original target: {args.epochs} epochs")
        if start_epoch >= args.epochs or (stale_epochs > 0 and stale_epochs >= args.patience):
            print("This run already finished or reached early stopping. Use --init-weights for a new schedule.")
            return
    elif args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location=device, weights_only=True))
        best_nmse = run_epoch(model, validation_data, device, scaler=scaler)["nmse_metric"]
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print("Loaded existing weights; optimizer and epoch count start fresh.")
    elif os.path.exists(args.weights_path) or os.path.exists(last_path):
        raise SystemExit("Existing checkpoints found. Use --resume, --init-weights, or a new --weights-path.")

    os.makedirs(os.path.dirname(os.path.abspath(args.weights_path)), exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    with open(os.path.join(args.logs_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    if best_state is not None:
        torch.save(best_state, args.weights_path + ".tmp")
        os.replace(args.weights_path + ".tmp", args.weights_path)
    csv_path = os.path.join(args.logs_dir, "training_log.csv")
    has_header = checkpoint is not None and os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with SummaryWriter(args.logs_dir) as writer, open(
        csv_path, "a" if checkpoint is not None else "w", newline="", encoding="utf-8"
    ) as handle:
        csv_writer = None
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            metrics = run_epoch(model, train_data, device, optimizer, scheduler, scaler)
            validation = run_epoch(model, validation_data, device, scaler=scaler)
            row = {"epoch": epoch, **metrics, **{f"val_{key}": value for key, value in validation.items()},
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            if csv_writer is None:
                csv_writer = csv.DictWriter(handle, fieldnames=list(row))
                if not has_header:
                    csv_writer.writeheader()
            csv_writer.writerow(row)
            handle.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            print(f"Validation NMSE: {validation['nmse_db_metric']:.2f} dB")
            if validation["nmse_metric"] < best_nmse:
                best_nmse, stale_epochs = validation["nmse_metric"], 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save(model.state_dict(), args.weights_path + ".tmp")
                os.replace(args.weights_path + ".tmp", args.weights_path)
            else:
                stale_epochs += 1
            train_data.on_epoch_end()
            numpy_rng = np.random.get_state()
            torch.save({
                "version": 1, "config": vars(args), "next_epoch": epoch + 1, "split_sha256": fingerprint,
                "model": model.state_dict(), "best_model": best_state,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "best_nmse": float(best_nmse), "stale_epochs": stale_epochs,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "python_rng": random.getstate(),
                "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            }, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)
            print(f"Resume checkpoint saved: {last_path}")
            if stale_epochs > 0 and stale_epochs >= args.patience:
                print("Early stopping")
                break
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    print(f"Best weights: {args.weights_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Resume uses the last completed epoch checkpoint, if one was saved.")
        raise SystemExit(130)
