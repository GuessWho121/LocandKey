"""Evaluate the verified epoch-59 model using its original, untouched test split."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

EXPECTED = {
    "weights/multiscenario_best.pt": "ab680be6239e13bed258bdc38a14a07c9d3ed3a088fad65a53db7cdf7eaf183c",
    "data/splits/split_summary.json": "9fe846ed4adc8767fb1a932047e83a3ca6384a1ef10ebce068ecbeeca2cba76a",
    "data/splits/split_indices.npz": "6057fd75393c5d103933ebafef8d1ad3a82b72f372a76dfb713b013ece27ed6c",
}


def evaluation_inputs(input_root):
    weights = list(input_root.glob("**/weights/multiscenario_best.pt"))
    if len(weights) != 1:
        raise ValueError(f"Expected one attached training output; found {weights}")
    source = weights[0].parent.parent
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Training artifact differs from the verified 60-epoch run: {name}")
    roots = {path.parent.parent for path in input_root.glob("**/channels.npy")}
    if len(roots) != 1:
        raise ValueError(f"Expected one CNS dataset; found {roots}")
    return source, roots.pop()


def main():
    source, dataset = evaluation_inputs(Path("/kaggle/input"))
    work = Path("/kaggle/working/evaluation")
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "generated").symlink_to(dataset, target_is_directory=True)
    print(f"Verified epoch-59 weights and original splits: {source}", flush=True)
    print("Evaluating all 14,024 test samples at nine SNRs (-10 to 30 dB)", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'GPU required'"],
                   check=True, env=env)
    subprocess.run([sys.executable, "-u", str(source / "eval.py"),
                    "--data-root", str(data), "--split-dir", str(source / "data/splits"),
                    "--weights-path", str(source / "weights/multiscenario_best.pt"),
                    "--output-dir", str(work / "results"), "--batch-size", "16", "--seed", "42"],
                   check=True, env=env)


if __name__ == "__main__":
    main()
