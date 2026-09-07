"""Train the LocandKey denoiser from memory-mapped multi-scenario CSI."""

import argparse
import json
import os

import tensorflow as tf

from model import build_resonance_model, nmse_db_metric, nmse_metric, resonance_loss
from preprocess import DATA_ROOT, SPLITS_DIR, make_csi_sequence


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(PROJECT_DIR, "weights", "locandkey_multiscenario_best.weights.h5")
DEFAULT_LOGS = os.path.join(PROJECT_DIR, "logs", "multiscenario")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multi-scenario CSI denoiser.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--split-dir", default=SPLITS_DIR)
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS)
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


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("--batch-size and --epochs must be positive")
    if args.snr_min_db >= args.snr_max_db:
        raise SystemExit("--snr-min-db must be lower than --snr-max-db")

    if args.smoke:
        args.epochs = 1
        args.max_train_samples = 64
        args.max_val_samples = 16
        args.weights_path = os.path.join(PROJECT_DIR, "weights", "locandkey_smoke.weights.h5")
        args.logs_dir = os.path.join(PROJECT_DIR, "logs", "smoke")

    tf.keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.require_gpu and not gpus:
        raise SystemExit("No TensorFlow GPU is available; omit --require-gpu only for smoke testing")
    if args.mixed_precision:
        if not gpus:
            raise SystemExit("--mixed-precision requires a GPU")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"Device: {gpus[0].name if gpus else 'CPU'}")

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

    model = build_resonance_model((128, 256, 2))
    total_steps = max(1, args.epochs * len(train_data))
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate,
        decay_steps=total_steps,
        alpha=args.min_learning_rate / args.learning_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=1.0),
        loss=resonance_loss(lambda_mag=0.05),
        metrics=[nmse_metric, nmse_db_metric],
    )
    print(f"Model parameters: {model.count_params():,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.weights_path)), exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    with open(os.path.join(args.logs_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.weights_path,
            monitor="val_nmse_metric",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_nmse_metric",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(args.logs_dir, "training_log.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=args.logs_dir, histogram_freq=0),
    ]
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=2,
        verbose=1,
    )
    print(f"Best weights: {args.weights_path}")


if __name__ == "__main__":
    main()
