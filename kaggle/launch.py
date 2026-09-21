"""Kaggle Script launcher; edit settings before submitting."""
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "0d210f23b370664d62d43111185ee284e1b1ea44"
MODULE_SOURCES = None  # Filled by build.py with the locally tested Python modules.
DATASET = Path("/kaggle/input/cns-proj")
WORK = Path("/kaggle/working/locandkey")
SMOKE = False
EPOCHS = 60
SOURCE_SPLITS = None  # Attached directory with original split JSON and NPZ.
HOLDOUT = None  # Optional: "city_4_phoenix_28"; requires original splits.
RESUME = None  # Full .last.pt path; requires matching splits and SMOKE=False.
PER_GPU_BATCH_SIZE = 4
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None


def dataset_root(input_root=Path("/kaggle/input")):
    roots = [path for path in (DATASET, DATASET / "generated", DATASET / "data" / "generated")
             if list(path.glob("*/channels.npy"))]
    if roots:
        return roots[0]
    matches = sorted(input_root.glob("**/channels.npy"))
    roots = sorted({match.parent.parent for match in matches})
    if len(roots) == 1:
        print(f"Using dataset mount: {roots[0]}")
        return roots[0]
    visible = sorted(str(path) for path in input_root.glob("*"))
    raise FileNotFoundError(
        "Attach CNS proj or set DATASET to the folder containing scenario folders. "
        f"Checked {DATASET}; candidates: {roots}; /kaggle/input contains: {visible}"
    )


def run(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], cwd=WORK, check=True)


def cuda_device_count():
    import torch
    return torch.cuda.device_count()


def main():
    if MODULE_SOURCES is None:
        raise RuntimeError("Run python kaggle/build.py, then push kaggle/bundle")
    dataset = dataset_root()
    if (HOLDOUT or RESUME) and SOURCE_SPLITS is None:
        raise ValueError("Holdout/resume requires SOURCE_SPLITS")
    if RESUME and SMOKE:
        raise ValueError("Set SMOKE=False when resuming")
    WORK.mkdir(parents=True, exist_ok=True)
    for name, source in MODULE_SOURCES.items():
        (WORK / name).write_text(source, encoding="utf-8")
    # Keep Kaggle's CUDA-enabled torch and numpy; do not install the laptop wheel.
    run("-m", "pip", "install", "tensorboard", "tqdm", "matplotlib")
    run("-c", "import torch; print(torch.__version__); "
        "assert torch.cuda.is_available(), 'Enable a Kaggle GPU'; "
        "assert hasattr(torch.amp, 'GradScaler'), 'Update the Kaggle runtime'; "
        "print(torch.cuda.device_count(), torch.cuda.get_device_name(0))")
    gpu_count = cuda_device_count()
    batch_size = PER_GPU_BATCH_SIZE * max(1, gpu_count)
    data = WORK / "data"
    data.mkdir(exist_ok=True)
    generated = data / "generated"
    if generated.is_symlink() and generated.resolve() == dataset.resolve():
        pass
    elif generated.exists() or generated.is_symlink():
        raise FileExistsError(generated)
    else:
        generated.symlink_to(dataset, target_is_directory=True)
    splits = data / "splits"
    if SOURCE_SPLITS is not None:
        splits.mkdir(exist_ok=True)
        for name in ("split_summary.json", "split_indices.npz"):
            source, target = Path(SOURCE_SPLITS) / name, splits / name
            if target.exists() and source.read_bytes() != target.read_bytes():
                raise ValueError(f"Existing splits differ: {target}")
            if not target.exists():
                shutil.copyfile(source, target)
    elif not (splits / "split_summary.json").exists():
        print("Creating NEW baseline splits; not verified against laptop splits.")
        run("preprocess.py", "--data-root", data, "--output-dir", splits)
    if HOLDOUT:
        output = splits / "holdout_phoenix"
        if not (output / "split_summary.json").exists():
            run("preprocess.py", "--data-root", data, "--source-split-dir", splits,
                "--holdout-scenario", HOLDOUT, "--output-dir", output)
        splits = output
    if RESUME:
        sys.path.insert(0, str(WORK))
        import torch
        from preprocess import split_fingerprint
        checkpoint = torch.load(RESUME, map_location="cpu", weights_only=True)
        if checkpoint.get("split_sha256") != split_fingerprint(splits):
            raise ValueError("Checkpoint does not match these exact split files")
    name = "holdout_phoenix" if HOLDOUT else "multiscenario"
    if SMOKE:
        name += "_smoke"
    extra = ["--max-train-samples", "64", "--max-val-samples", "16"] if SMOKE else []
    if not SMOKE:
        for flag, value in (("--max-train-samples", MAX_TRAIN_SAMPLES), ("--max-val-samples", MAX_VAL_SAMPLES)):
            if value is not None:
                extra += [flag, str(value)]
    if RESUME:
        extra += ["--resume", RESUME]
    (WORK / "kaggle_run.json").write_text(json.dumps({
        "revision": REVISION, "dataset": str(dataset), "smoke": SMOKE,
        "holdout": HOLDOUT, "source_splits": str(SOURCE_SPLITS),
        "gpu_count": gpu_count, "per_gpu_batch_size": PER_GPU_BATCH_SIZE,
        "batch_size": batch_size, "epochs": 1 if SMOKE else EPOCHS,
        "module_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in MODULE_SOURCES.items()},
        "max_train_samples": MAX_TRAIN_SAMPLES, "max_val_samples": MAX_VAL_SAMPLES,
        "parallelism": "DistributedDataParallel" if gpu_count > 1 else "single",
    }, indent=2), encoding="utf-8")
    command = (["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpu_count}"]
               if gpu_count > 1 else [])
    run(*command, "train.py", "--data-root", data, "--split-dir", splits,
        "--weights-path", WORK / "weights" / f"{name}_best.pt",
        "--logs-dir", WORK / "logs" / name, "--epochs", "1" if SMOKE else str(EPOCHS),
        "--patience", "10", "--batch-size", str(batch_size), "--seed", "42",
        "--mixed-precision", "--require-gpu", *extra)


if __name__ == "__main__":
    main()
