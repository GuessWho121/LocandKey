"""Fresh Phoenix holdout training followed by matched baseline evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SPLIT_SHA256 = "741a6abc5a918e7694682f6bb566846bef548c8b240a6b5d46e6d3bb108eb9e6"
WEIGHTS_SHA256 = "ab680be6239e13bed258bdc38a14a07c9d3ed3a088fad65a53db7cdf7eaf183c"
SCENARIO = "city_4_phoenix_28"


def verify_source(source):
    splits = source / "data/splits"
    digest = hashlib.sha256(b"".join((splits / name).read_bytes()
                                   for name in ("split_summary.json", "split_indices.npz"))).hexdigest()
    if digest != SPLIT_SHA256:
        raise ValueError("Attached training output has different original splits")
    if hashlib.sha256((source / "weights/multiscenario_best.pt").read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Attached baseline differs from the verified epoch-59 model")
    manifest = json.loads((source / "kaggle_run.json").read_text())
    for name in ("model.py", "preprocess.py", "train.py", "eval.py"):
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != manifest["module_sha256"][name]:
            raise ValueError(f"Training module hash mismatch: {name}")


def main():
    root = Path("/kaggle/input")
    weights = list(root.glob("**/weights/multiscenario_best.pt"))
    datasets = {path.parent.parent for path in root.glob("**/channels.npy")}
    if len(weights) != 1 or len(datasets) != 1:
        raise ValueError(f"Expected one baseline and CNS dataset: {weights}, {datasets}")
    source = weights[0].parent.parent
    verify_source(source)
    work = Path("/kaggle/working/holdout_phoenix")
    data, splits = work / "data", work / "data/splits"
    data.mkdir(parents=True, exist_ok=True)
    (data / "generated").symlink_to(datasets.pop(), target_is_directory=True)
    for name in ("model.py", "preprocess.py", "train.py", "eval.py"):
        shutil.copyfile(source / name, work / name)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    def run(*args):
        subprocess.run([sys.executable, "-u", *map(str, args)], cwd=work, env=env, check=True)
    run("preprocess.py", "--data-root", data, "--source-split-dir", source / "data/splits",
        "--holdout-scenario", SCENARIO, "--output-dir", splits)
    summary = json.loads((splits / "split_summary.json").read_text())
    counts = summary["scenarios"][SCENARIO]["splits"]
    if counts["train"] or counts["validation"] or summary["source_split_sha256"] != SPLIT_SHA256:
        raise ValueError("Phoenix isolation or source split integrity failed")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    gpus = torch.cuda.device_count()
    parallel = ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else []
    best = work / "weights/holdout_phoenix_best.pt"
    print(f"Fresh holdout training: {summary['totals']}; GPUs={gpus}", flush=True)
    run(*parallel, "train.py", "--data-root", data, "--split-dir", splits,
        "--weights-path", best, "--logs-dir", work / "logs", "--epochs", "60",
        "--patience", "10", "--batch-size", gpus * 4, "--seed", "42", "--mixed-precision", "--require-gpu")
    run("eval.py", "--data-root", data, "--split-dir", splits, "--weights-path", best,
        "--baseline-weights", weights[0], "--output-dir", work / "evaluation", "--batch-size", "16", "--seed", "42")


if __name__ == "__main__":
    main()
