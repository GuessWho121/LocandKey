"""
Resonance - Data Preprocessor v2
==================================
Loads channels_o1_28.npy  (N, 128, 256, 2)  float32
Fixes NaNs/Infs, normalizes, augments, and saves train/val/test splits.

Run:
    python preprocess_2.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
INPUT_NPY      = r"D:\ML Models\scenarios\channels_o1_28.npy"
OUTPUT_DIR     = r"D:\ML Models\scenarios\splits"

TARGET_H       = 128    # antennas
TARGET_W       = 256    # subcarriers

NUM_AUGMENTS   = 0      # augmented copies per original sample
NOISE_LEVEL    = 0.05   # std of augmentation noise

TEST_SIZE      = 0.15
VAL_SIZE       = 0.15   # fraction of remaining after test split
RANDOM_SEED    = 42
# ══════════════════════════════════════════════════════


def load_and_validate(path):
    print(f"\n[1/5] Loading data...")
    print(f"      Path: {path}")

    raw = np.load(path, allow_pickle=False)   # plain float32, no pickle needed

    print(f"      Raw shape : {raw.shape}  dtype: {raw.dtype}")

    # Accept either (N, A, S, 2) or (N, A, S) complex
    if raw.ndim == 3 and np.iscomplexobj(raw):
        print(f"      Converting complex → I/Q...")
        raw = np.stack([np.real(raw), np.imag(raw)], axis=-1).astype(np.float32)
    elif raw.ndim == 4 and raw.shape[-1] == 2:
        raw = raw.astype(np.float32)
    else:
        raise ValueError(f"Unexpected shape {raw.shape}. Expected (N,A,S,2) or (N,A,S) complex.")

    return raw


def fix_nans(data):
    print(f"\n[2/5] Checking data quality...")

    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    print(f"      NaNs : {nan_count:,}")
    print(f"      Infs : {inf_count:,}")

    if nan_count > 0 or inf_count > 0:
        # Replace NaN/Inf with 0 — these are users with no signal (blocked paths)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"      ✅ Replaced NaN/Inf with 0")

    # Drop samples that are entirely zero (users with no paths at all)
    energy = np.sum(data**2, axis=(1, 2, 3))
    valid  = energy > 1e-10
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        data = data[valid]
        print(f"      Dropped {n_dropped:,} zero-energy samples")

    print(f"      Clean samples: {data.shape[0]:,}")
    return data


def resize(data, target_h, target_w):
    """Pad or truncate spatial dims to (N, target_h, target_w, 2)."""
    _, h, w, _ = data.shape
    if h == target_h and w == target_w:
        return data

    print(f"\n      Resizing {h}×{w} → {target_h}×{target_w}")
    data = data[:, :target_h, :target_w, :]
    ph   = max(0, target_h - data.shape[1])
    pw   = max(0, target_w - data.shape[2])
    if ph or pw:
        data = np.pad(data, ((0,0),(0,ph),(0,pw),(0,0)))
    return data


def normalize(data):
    """
    Per-sample L2 normalization on the complex channel.
    Done on the combined I/Q magnitude so phase is preserved.
    """
    print(f"\n[3/5] Normalizing...")

    # Compute per-sample norm: sqrt(sum of I^2 + Q^2)
    norm = np.sqrt(np.sum(data**2, axis=(1, 2, 3), keepdims=True))

    # Avoid division by zero (already removed zero samples, but just in case)
    norm = np.maximum(norm, 1e-10)
    data = data / norm

    print(f"      ✅ Per-sample L2 normalization applied")
    print(f"      Sample norm check (should be ~1.0): "
          f"{np.sqrt(np.sum(data[0]**2)):.4f}")

    return data


def augment(data, num_augments, noise_level):
    """
    Augmentation strategies:
      1. Additive Gaussian noise (simulates varying SNR)
      2. Random amplitude scaling ±15% (simulates path loss variation)
      3. Subcarrier axis flip (frequency symmetry)
      4. Antenna axis flip (spatial symmetry)
    """
    print(f"\n[4/5] Augmenting  ({num_augments}× per sample)...")

    augmented = [data]   # keep originals

    for i in range(num_augments):
        batch = data.copy()

        if i % 3 == 0:
            # Strategy 1: AWGN noise
            noise  = np.random.normal(0, noise_level, batch.shape).astype(np.float32)
            batch  = batch + noise

        elif i % 3 == 1:
            # Strategy 2: Amplitude scaling + light noise
            scale  = np.random.uniform(0.85, 1.15, (len(batch),1,1,1)).astype(np.float32)
            noise  = np.random.normal(0, noise_level*0.5, batch.shape).astype(np.float32)
            batch  = batch * scale + noise

        else:
            # Strategy 3: Spatial flip along subcarrier axis
            batch  = np.flip(batch, axis=2).copy()
            noise  = np.random.normal(0, noise_level*0.3, batch.shape).astype(np.float32)
            batch  = batch + noise

        augmented.append(batch.astype(np.float32))

    result = np.concatenate(augmented, axis=0)

    # Shuffle so augmented copies aren't all grouped together
    idx    = np.random.permutation(len(result))
    result = result[idx]

    print(f"      Original : {data.shape[0]:>7,} samples")
    print(f"      Augmented: {result.shape[0]:>7,} samples")
    return result


def split_and_save(data, output_dir, test_size, val_size, seed):
    print(f"\n[5/5] Splitting and saving...")
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_temp  = train_test_split(data,   test_size=test_size, random_state=seed)
    X_val,   X_test  = train_test_split(X_temp, test_size=0.5,       random_state=seed)

    splits = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test}

    for name, arr in splits.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        mb   = os.path.getsize(path) / 1e6
        print(f"      {name:8s}: {arr.shape[0]:>7,} samples  {arr.shape}  {mb:.1f} MB  → {path}")

    return splits


def main():
    print("\n" + "═"*60)
    print("  Resonance — Data Preprocessor")
    print("═"*60)

    data = load_and_validate(INPUT_NPY)
    data = fix_nans(data)
    data = resize(data, TARGET_H, TARGET_W)
    data = normalize(data)
    data = augment(data, NUM_AUGMENTS, NOISE_LEVEL)
    splits = split_and_save(data, OUTPUT_DIR, TEST_SIZE, VAL_SIZE, RANDOM_SEED)

    print(f"\n{'═'*60}")
    print(f"  ✅  Preprocessing complete!")
    print(f"  Train : {splits['X_train'].shape}")
    print(f"  Val   : {splits['X_val'].shape}")
    print(f"  Test  : {splits['X_test'].shape}")
    print(f"  Input shape for model: {splits['X_train'].shape[1:]}")
    print(f"{'═'*60}\n")
    print(f"  Next step → update model.py then run train.py")


if __name__ == "__main__":
    main()