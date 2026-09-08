"""Run with python test_pytorch.py; synthetic data stays in a temporary folder."""

import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from model import build_resonance_model, nmse_metric, resonance_loss
from preprocess import create_splits, create_holdout_splits, make_csi_sequence, split_fingerprint
from eval import holdout_decisions


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        rng = np.random.default_rng(42)
        names = ("asu_campus_3p5", "city_16_sanfrancisco_28", "city_4_phoenix_28", "i1_2p4", "o1_28")
        for name in names:
            scenario = work / "generated" / name
            scenario.mkdir(parents=True)
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
        assert len(report["results"]) == 12
        assert all(np.isfinite(row["model_nmse_db"]) for row in report["results"])
        assert (work / "evaluation" / "nmse_vs_snr.png").is_file()
        common = ["--data-root", str(work), "--split-dir", str(splits), "--epochs", "2",
                  "--batch-size", "1", "--max-train-samples", "2", "--max-val-samples", "1"]
        uninterrupted = work / "uninterrupted.pt"
        resumed = work / "resumed.pt"
        subprocess.run([sys.executable, str(root / "train.py"), *common,
                        "--weights-path", str(uninterrupted), "--logs-dir", str(work / "baseline_logs")],
                       check=True, env=env)
        interrupt_code = """
import train
original = train.run_epoch
calls = 0
def interrupted(*args, **kwargs):
    global calls
    calls += 1
    if calls == 3:
        raise KeyboardInterrupt
    return original(*args, **kwargs)
train.run_epoch = interrupted
try:
    train.main()
except KeyboardInterrupt:
    pass
"""
        resume_args = [*common, "--weights-path", str(resumed), "--logs-dir", str(work / "resume_logs")]
        subprocess.run([sys.executable, "-c", interrupt_code, *resume_args], cwd=root, check=True, env=env)
        saved = torch.load(str(resumed) + ".last.pt", weights_only=True)
        assert saved["next_epoch"] == 1 and saved["optimizer"]["state"]
        subprocess.run([sys.executable, str(root / "train.py"), *resume_args, "--resume"], check=True, env=env)
        actual = torch.load(str(resumed) + ".last.pt", weights_only=True)
        expected = torch.load(str(uninterrupted) + ".last.pt", weights_only=True)
        assert actual["next_epoch"] == 2 and actual["scheduler"] == expected["scheduler"]
        for key in expected["model"]:
            torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
        with open(work / "resume_logs" / "training_log.csv", newline="") as handle:
            assert [row["epoch"] for row in csv.DictReader(handle)] == ["0", "1"]
        subprocess.run([sys.executable, str(root / "train.py"), *common,
                        "--weights-path", str(work / "warm.pt"), "--logs-dir", str(work / "warm_logs"),
                        "--init-weights", str(weights)], check=True, env=env)
        assert (work / "warm.pt.last.pt").is_file()
        holdout = "city_4_phoenix_28"
        experiment = work / "splits" / "holdout_phoenix"
        source_hash = split_fingerprint(splits)
        create_holdout_splits(str(work), str(splits), str(experiment), holdout)
        assert split_fingerprint(splits) == source_hash
        with np.load(splits / "split_indices.npz") as source, np.load(experiment / "split_indices.npz") as target_indices:
            for name in names:
                for split in ("train", "validation", "test"):
                    if name != holdout:
                        np.testing.assert_array_equal(source[f"{split}__{name}"], target_indices[f"{split}__{name}"])
                parts = [target_indices[f"{split}__{name}"] for split in ("train", "validation", "test")]
                assert len(np.unique(np.concatenate(parts))) == sum(map(len, parts))
            assert len(target_indices[f"train__{holdout}"]) == len(target_indices[f"validation__{holdout}"]) == 0
            np.testing.assert_array_equal(np.sort(target_indices[f"test__{holdout}"]), np.arange(10))
            np.testing.assert_array_equal(target_indices[f"original_test__{holdout}"], source[f"test__{holdout}"])
        for split in ("train", "validation"):
            sequence = make_csi_sequence(str(work), str(experiment), split, 1)
            assert all(sequence.scenario_names[int(ref[0])] != holdout for ref in sequence.references)
            assert sequence[0][0].shape == (1, 128, 256, 2)
            for channel in sequence.channels.values():
                channel._mmap.close()
        for source_dir, output_dir, name in ((splits, splits, holdout), (splits, experiment, holdout),
                                              (splits, work / "missing", "absent")):
            try:
                create_holdout_splits(str(work), str(source_dir), str(output_dir), name)
            except (ValueError, FileExistsError):
                pass
            else:
                raise AssertionError("Invalid holdout request accepted")
        for invalid in ("overlap", "out_of_range", "float", "missing_scenario"):
            bad = work / invalid
            bad.mkdir()
            shutil.copyfile(splits / "split_summary.json", bad / "split_summary.json")
            with np.load(splits / "split_indices.npz") as source:
                arrays = {key: source[key].copy() for key in source.files}
            if invalid == "overlap":
                arrays[f"train__{holdout}"][0] = arrays[f"test__{holdout}"][0]
            elif invalid == "out_of_range":
                arrays[f"train__{holdout}"][0] = 999
            elif invalid == "float":
                arrays[f"train__{holdout}"] = arrays[f"train__{holdout}"].astype(float)
            else:
                del arrays[f"test__{holdout}"]
            np.savez(bad / "split_indices.npz", **arrays)
            try:
                create_holdout_splits(str(work), str(bad), str(work / (invalid + "_output")), holdout)
            except (ValueError, KeyError):
                pass
            else:
                raise AssertionError(f"Invalid indices accepted: {invalid}")
        holdout_weights = work / "holdout.pt"
        holdout_args = ["--data-root", str(work), "--split-dir", str(experiment), "--epochs", "1",
                        "--batch-size", "1", "--max-train-samples", "2", "--max-val-samples", "1",
                        "--weights-path", str(holdout_weights), "--logs-dir", str(work / "holdout_logs")]
        for flags in (("--init-weights", str(weights)), ("--resume", str(weights) + ".last.pt")):
            result = subprocess.run([sys.executable, str(root / "train.py"), *holdout_args, *flags],
                                    capture_output=True, text=True, env=env)
            assert result.returncode != 0 and ("not allowed" in result.stderr or "does not belong" in result.stderr)
        subprocess.run([sys.executable, str(root / "train.py"), *holdout_args], check=True, env=env)
        subprocess.run([sys.executable, str(root / "eval.py"), "--data-root", str(work),
                        "--split-dir", str(experiment), "--weights-path", str(holdout_weights),
                        "--baseline-weights", str(holdout_weights), "--batch-size", "1",
                        "--output-dir", str(work / "holdout_eval")], check=True, env=env)
        holdout_report = json.loads((work / "holdout_eval" / "evaluation_summary.json").read_text())
        assert len(holdout_report["groups"]) == 18 and len(holdout_report["comparisons"]) == 45
        assert all(row["samples"] == 2 for row in holdout_report["original_holdout_test"])
        assert all(row["samples"] == 10 for row in holdout_report["groups"] if row["scenario"] == "unseen")
        assert all(row["samples"] == 8 for row in holdout_report["groups"] if row["scenario"] == "seen")
        assert all(row["degradation_db"] == 0 for row in holdout_report["comparisons"])
        assert (work / "holdout_eval" / "holdout_nmse_vs_snr.png").is_file()
        decisions = holdout_decisions([{"snr_db": 0, "nmse_gain_db": 3}, {"snr_db": 10, "nmse_gain_db": -1}],
                                     [{"scope": "seen", "snr_db": 10, "degradation_db": 1.1}])
        assert decisions["unseen_gain_at_least_2_db"] is False
        assert decisions["unseen_worsened_snrs"] == [10] and len(decisions["seen_degradation_over_1_db"]) == 1
        assert holdout_decisions([], [])["unseen_gain_at_least_2_db"] is None
    print("PASS: training/resume, holdout isolation, invalid splits, matched evaluation and decision rules")


if __name__ == "__main__":
    main()
