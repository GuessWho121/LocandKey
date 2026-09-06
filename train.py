"""
Resonance - Training Engine v2
================================
Loads preprocessed splits, builds the upgraded model,
and trains with cosine annealing + on-the-fly augmentation.

Run:
    python train.py
"""

import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import os

from model import build_resonance_model, resonance_loss, nmse_metric, nmse_db_metric

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
SPLITS_DIR    = r"D:\ML Models\scenarios\splits"
WEIGHTS_DIR   = "weights"
LOGS_DIR      = "logs"

BATCH_SIZE    = 4      # lower than before — model is bigger now
EPOCHS        = 60
LR_MAX = 3e-4   # was 1e-3    # peak learning rate for cosine schedule
LR_MIN        = 1e-6    # floor
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0     # gradient clipping norm

# Loss weights
LAMBDA_PHYS = 0.05   # was 0.10
LAMBDA_SPEC = 0.05   # was 0.10
LAMBDA_MAG  = 0.02   # was 0.05

# On-the-fly augmentation noise std
AUG_NOISE_STD = 0.05
# ══════════════════════════════════════════════════════


# ── Hardware setup ────────────────────────────────────
print("⚙️  Hardware setup...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"   GPU: {gpus[0].name}")
else:
    print("   No GPU found — training on CPU")

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print(f"   Mixed precision: {policy.compute_dtype}")


# ── Data loading ──────────────────────────────────────
print("\n📂 Loading splits...")

X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))

print(f"   Train : {X_train.shape}  {X_train.nbytes/1e6:.1f} MB")
print(f"   Val   : {X_val.shape}")

input_shape = X_train.shape[1:]   # (128, 256, 2)
print(f"   Input shape: {input_shape}")


# ── On-the-fly augmentation ───────────────────────────
def augment_batch(x_clean):
    """
    Add random Gaussian noise to create noisy input on-the-fly.
    This is the actual denoising task: model sees noisy, predicts clean.
    Noise std is randomised per batch to train across a range of SNRs.
    """
    noise_std = tf.random.uniform([], minval=0.02, maxval=AUG_NOISE_STD)
    noise     = tf.random.normal(tf.shape(x_clean), stddev=noise_std)
    x_noisy   = x_clean + noise
    return x_noisy, x_clean   # (noisy input, clean target)


def make_dataset(X, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(X)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 5000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, BATCH_SIZE, shuffle=True)
val_ds   = make_dataset(X_val,   BATCH_SIZE, shuffle=False)


# ── Model ─────────────────────────────────────────────
print("\n🏗️  Building model...")
model = build_resonance_model(input_shape=input_shape)
print(f"   Parameters: {model.count_params():,}")


# ── Cosine annealing LR schedule ─────────────────────
steps_per_epoch = len(X_train) // BATCH_SIZE
total_steps     = EPOCHS * steps_per_epoch

class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine annealing: smoothly decays LR from LR_MAX → LR_MIN.
    Much better than ReduceLROnPlateau for ConvNeXt — avoids sudden drops
    that can destabilise the LayerNorm layers.
    """
    def __init__(self, lr_max, lr_min, total_steps, warmup_steps=200):
        self.lr_max       = lr_max
        self.lr_min       = lr_min
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step    = tf.cast(step, tf.float32)
        # Linear warmup
        warmup  = self.lr_max * (step / self.warmup_steps)
        # Cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine   = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                   1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {'lr_max': self.lr_max, 'lr_min': self.lr_min,
                'total_steps': self.total_steps, 'warmup_steps': self.warmup_steps}

lr_schedule = CosineAnnealingSchedule(LR_MAX, LR_MIN, total_steps, warmup_steps=200)


# ── Compile ───────────────────────────────────────────
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    global_clipnorm=GRAD_CLIP
)

model.compile(
    optimizer=optimizer,
    loss=resonance_loss(
        lambda_phys=LAMBDA_PHYS,
        lambda_spec=LAMBDA_SPEC,
        lambda_mag=LAMBDA_MAG
    ),
    metrics=[nmse_metric, nmse_db_metric]
)
print("   ✅ Model compiled")


# ── Directories ───────────────────────────────────────
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)


# ── Callbacks ─────────────────────────────────────────
callbacks = [
    # Save best weights by validation NMSE
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(WEIGHTS_DIR, "resonance_best.weights.h5"),
        monitor="val_nmse_metric",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
    ),
    # Early stopping on val NMSE
    tf.keras.callbacks.EarlyStopping(
        monitor="val_nmse_metric",
        patience=12,
        restore_best_weights=True,
        mode="min",
        verbose=1
    ),
    # TensorBoard — run: tensorboard --logdir logs
    tf.keras.callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=0,
        update_freq="epoch"
    ),
    # CSV log — easy to plot later
    tf.keras.callbacks.CSVLogger(
        os.path.join(LOGS_DIR, "training_log.csv"),
        append=False
    )
]


# ── Training ──────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🚀 Training for up to {EPOCHS} epochs...")
    print(f"   Steps/epoch : {steps_per_epoch}")
    print(f"   Total steps : {total_steps}")
    print(f"   Batch size  : {BATCH_SIZE}")
    print(f"   LR schedule : cosine {LR_MAX} → {LR_MIN}\n")

    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        # Print final metrics
        best_val_nmse = min(history.history.get("val_nmse_metric", [float('inf')]))
        best_val_db   = min(history.history.get("val_nmse_db_metric", [float('inf')]))
        print(f"\n{'═'*50}")
        print(f"  Training complete!")
        print(f"  Best val NMSE    : {best_val_nmse:.6f}")
        print(f"  Best val NMSE dB : {best_val_db:.2f} dB")
        print(f"  Weights saved to : {WEIGHTS_DIR}/resonance_best.weights.h5")
        print(f"{'═'*50}\n")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted. Weights in 'weights/' are intact.")