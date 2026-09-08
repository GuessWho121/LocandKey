"""Build reproducible multi-scenario splits without copying CSI arrays."""

import argparse
import json
import math
import os

import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LOCANDKEY_DATA_ROOT", os.path.join(PROJECT_DIR, "data"))
SPLITS_DIR = os.environ.get("LOCANDKEY_SPLITS_DIR", os.path.join(DATA_ROOT, "splits"))
EXPECTED_SAMPLE_SHAPE = (128, 256, 2)


def discover_channels(data_root):
    generated = os.path.join(data_root, "generated")
    if not os.path.isdir(generated):
        raise FileNotFoundError(f"Generated data directory not found: {generated}")
    channels = {}
    for entry in sorted(os.scandir(generated), key=lambda item: item.name):
        path = os.path.join(entry.path, "channels.npy")
        if entry.is_dir() and os.path.isfile(path):
            channels[entry.name] = path
    if not channels:
        raise FileNotFoundError(f"No scenario channels.npy files found under {generated}")
    return channels


def validate_channels(path, chunk_size):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 4 or data.shape[1:] != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(f"Unexpected shape {data.shape} in {path}")
    if data.dtype != np.float32:
        raise ValueError(f"Unexpected dtype {data.dtype} in {path}; expected float32")

    valid = []
    nonfinite_count = 0
    zero_count = 0
    for start in range(0, len(data), chunk_size):
        chunk = np.asarray(data[start:start + chunk_size])
        finite = np.isfinite(chunk).all(axis=(1, 2, 3))
        energy = np.sum(chunk * chunk, axis=(1, 2, 3), dtype=np.float64)
        nonfinite_count += int((~finite).sum())
        zero_count += int((finite & (energy == 0)).sum())
        keep = finite & (energy > 0)
        valid.append(np.arange(start, start + len(chunk), dtype=np.int64)[keep])

    if nonfinite_count:
        raise ValueError(f"{path} contains {nonfinite_count} samples with NaN or Inf")
    return data, np.concatenate(valid), zero_count


def create_splits(data_root, output_dir, seed=42, train_ratio=0.70, val_ratio=0.15, chunk_size=32):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must be positive and leave room for the test split")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    split_arrays = {}
    summary = {
        "version": 1,
        "seed": seed,
        "ratios": {"train": train_ratio, "validation": val_ratio, "test": 1 - train_ratio - val_ratio},
        "sample_shape": list(EXPECTED_SAMPLE_SHAPE),
        "scenarios": {},
        "totals": {"source": 0, "usable": 0, "train": 0, "validation": 0, "test": 0},
    }

    for scenario_number, (scenario, path) in enumerate(discover_channels(data_root).items()):
        print(f"Validating {scenario}: {path}")
        data, valid_indices, zero_count = validate_channels(path, chunk_size)
        indices = np.random.default_rng(seed + scenario_number).permutation(valid_indices)
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)
        scenario_splits = {
            "train": indices[:train_end],
            "validation": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        assert not np.intersect1d(scenario_splits["train"], scenario_splits["validation"]).size
        assert not np.intersect1d(scenario_splits["train"], scenario_splits["test"]).size
        assert not np.intersect1d(scenario_splits["validation"], scenario_splits["test"]).size

        counts = {name: len(values) for name, values in scenario_splits.items()}
        for name, values in scenario_splits.items():
            split_arrays[f"{name}__{scenario}"] = values
            summary["totals"][name] += counts[name]
        summary["totals"]["source"] += len(data)
        summary["totals"]["usable"] += len(indices)
        summary["scenarios"][scenario] = {
            "path": os.path.relpath(path, data_root).replace("\\", "/"),
            "source_samples": len(data),
            "usable_samples": len(indices),
            "dropped_zero_energy": zero_count,
            "splits": counts,
        }
        print(f"  usable={len(indices):,} train={counts['train']:,} val={counts['validation']:,} test={counts['test']:,}")

    os.makedirs(output_dir, exist_ok=True)
    indices_path = os.path.join(output_dir, "split_indices.npz")
    summary_path = os.path.join(output_dir, "split_summary.json")
    indices_tmp = indices_path + ".tmp"
    summary_tmp = summary_path + ".tmp"
    with open(indices_tmp, "wb") as handle:
        np.savez_compressed(handle, **split_arrays)
    with open(summary_tmp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(indices_tmp, indices_path)
    os.replace(summary_tmp, summary_path)
    print(f"Saved split indices: {indices_path}")
    print(f"Saved split summary: {summary_path}")
    print(f"Totals: {summary['totals']}")
    return summary


def load_split_files(data_root, split_dir):
    with open(os.path.join(split_dir, "split_summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    indices = np.load(os.path.join(split_dir, "split_indices.npz"), allow_pickle=False)
    channels = {
        name: np.load(os.path.join(data_root, details["path"]), mmap_mode="r", allow_pickle=False)
        for name, details in summary["scenarios"].items()
    }
    return summary, indices, channels


def normalize_batch(batch):
    batch = np.asarray(batch, dtype=np.float32)
    norm = np.sqrt(np.sum(batch * batch, axis=(1, 2, 3), keepdims=True, dtype=np.float64)).astype(np.float32)
    return batch / np.maximum(norm, np.float32(1e-10))


def add_awgn(batch, snr_db, rng):
    snr_db = np.asarray(snr_db, dtype=np.float32).reshape(-1, 1, 1, 1)
    power = np.mean(batch[..., 0] ** 2 + batch[..., 1] ** 2, axis=(1, 2), keepdims=True)[..., None]
    noise_std = np.sqrt(power / (2.0 * np.power(10.0, snr_db / 10.0)))
    noise = rng.normal(size=batch.shape).astype(np.float32) * noise_std.astype(np.float32)
    return batch + noise


def make_csi_sequence(data_root, split_dir, split, batch_size, seed=42, shuffle=False,
                      snr_min_db=-10.0, snr_max_db=30.0, fixed_snr_db=None, max_samples=None):
    if batch_size <= 0 or (max_samples is not None and max_samples <= 0):
        raise ValueError("batch_size and max_samples must be positive")

    summary, indices_file, channels = load_split_files(data_root, split_dir)
    scenario_names = list(summary["scenarios"])
    references = []
    for scenario_id, scenario in enumerate(scenario_names):
        values = indices_file[f"{split}__{scenario}"]
        references.append(np.column_stack((np.full(len(values), scenario_id, dtype=np.int64), values)))
    indices_file.close()
    references = np.concatenate(references)
    if max_samples and max_samples < len(references):
        selected = np.random.default_rng(seed).permutation(len(references))[:max_samples]
        references = references[selected]

    if not len(references):
        raise ValueError(f"The {split} split is empty")

    class CSISequence:
        def __init__(self):
            self.references = references
            self.channels = channels
            self.scenario_names = scenario_names
            self.batch_size = batch_size
            self.seed = seed
            self.shuffle = shuffle
            self.epoch = 0
            self.order = np.arange(len(references))
            if shuffle:
                self.order = np.random.default_rng(seed).permutation(self.order)

        def __len__(self):
            return math.ceil(len(self.references) / self.batch_size)

        def __getitem__(self, batch_number):
            if batch_number < 0 or batch_number >= len(self):
                raise IndexError(batch_number)
            positions = self.order[batch_number * self.batch_size:(batch_number + 1) * self.batch_size]
            batch_refs = self.references[positions]
            clean = np.empty((len(batch_refs),) + EXPECTED_SAMPLE_SHAPE, dtype=np.float32)
            for scenario_id in np.unique(batch_refs[:, 0]):
                destinations = np.where(batch_refs[:, 0] == scenario_id)[0]
                scenario = self.scenario_names[int(scenario_id)]
                clean[destinations] = self.channels[scenario][batch_refs[destinations, 1]]
            clean = normalize_batch(clean)
            noise_epoch = self.epoch if fixed_snr_db is None else 0
            rng = np.random.default_rng(self.seed + noise_epoch * 1_000_003 + batch_number)
            snr = (rng.uniform(snr_min_db, snr_max_db, len(clean)) if fixed_snr_db is None
                   else np.full(len(clean), fixed_snr_db))
            return add_awgn(clean, snr, rng), clean

        def on_epoch_end(self):
            self.epoch += 1
            if self.shuffle:
                self.order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.references))

    return CSISequence()


def parse_args():
    parser = argparse.ArgumentParser(description="Create memory-mapped multi-scenario CSI splits.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-dir", default=SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_splits(args.data_root, args.output_dir, args.seed, args.train_ratio, args.val_ratio, args.chunk_size)
