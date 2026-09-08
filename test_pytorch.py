"""Run with python test_pytorch.py; synthetic data stays in a temporary folder."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from model import build_resonance_model, nmse_metric, resonance_loss
from preprocess import create_splits, make_csi_sequence


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        scenario = work / "generated" / "synthetic"
        scenario.mkdir(parents=True)
        rng = np.random.default_rng(42)
        np.save(scenario / "channels.npy", rng.normal(size=(10, 128, 256, 2)).astype(np.float32))
        splits = work / "splits"
        create_splits(str(work), str(splits))
        data = make_csi_sequence(str(work), str(splits), "validation", 1, fixed_snr_db=10)
        noisy, clean = data[0]
        data.on_epoch_end()
        np.testing.assert_array_equal(noisy, data[0][0])
        np.testing.assert_allclose(np.sum(clean * clean), 1, rtol=1e-5)
        target = torch.from_numpy(clean).permute(0, 3, 1, 2)
        prediction = target * 0.5
        expected = np.mean((clean - clean * 0.5) ** 2) / (np.mean(clean ** 2) + 1e-8)
        np.testing.assert_allclose(nmse_metric(target, prediction).item(), expected, rtol=1e-5)
        assert resonance_loss()(target, target).item() == 0
        for channel in data.channels.values():
            channel._mmap.close()
        model = build_resonance_model()
        output = model(target)
        assert output.shape == target.shape and output.dtype == torch.float32
        resonance_loss()(target, output).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        weights = work / "test.pt"
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg")
        subprocess.run([sys.executable, str(root / "train.py"), "--data-root", str(work),
                        "--split-dir", str(splits), "--weights-path", str(weights),
                        "--logs-dir", str(work / "logs"), "--epochs", "1", "--batch-size", "1",
                        "--max-train-samples", "2", "--max-val-samples", "1"], check=True, env=env)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        with open(work / "logs" / "training_log.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1 and np.isfinite(float(rows[0]["val_loss"]))
        subprocess.run([sys.executable, str(root / "eval.py"), "--data-root", str(work),
                        "--split-dir", str(splits), "--weights-path", str(weights),
                        "--output-dir", str(work / "evaluation"), "--batch-size", "1",
                        "--max-samples-per-scenario", "1", "--snr-db", "0", "10"], check=True, env=env)
        report = json.loads((work / "evaluation" / "evaluation_summary.json").read_text())
        assert len(report["results"]) == 4
        assert all(np.isfinite(row["model_nmse_db"]) for row in report["results"])
        assert (work / "evaluation" / "nmse_vs_snr.png").is_file()
    print("PASS: layout, loss, gradients, deterministic validation, training, checkpoint and evaluation")


if __name__ == "__main__":
    main()
