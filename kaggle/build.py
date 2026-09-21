"""Bundle local modules into Kaggle's single uploaded script. No network access."""
import argparse
from pathlib import Path
import shutil


def build(output, *, smoke=False, epochs=60, max_train_samples=None, max_val_samples=None):
    folder = Path(__file__).resolve().parent
    modules = {name: (folder.parent / name).read_text(encoding="utf-8")
               for name in ("model.py", "preprocess.py", "train.py", "eval.py")}
    for name, source in modules.items():
        compile(source, name, "exec")
    source = (folder / "launch.py").read_text(encoding="utf-8")
    for old, new in {
        "MODULE_SOURCES = None": f"MODULE_SOURCES = {modules!r}",
        "SMOKE = False": f"SMOKE = {smoke!r}",
        "EPOCHS = 60": f"EPOCHS = {epochs!r}",
        "MAX_TRAIN_SAMPLES = None": f"MAX_TRAIN_SAMPLES = {max_train_samples!r}",
        "MAX_VAL_SAMPLES = None": f"MAX_VAL_SAMPLES = {max_val_samples!r}",
    }.items():
        if source.count(old) != 1:
            raise ValueError(f"Launcher setting changed: {old}")
        source = source.replace(old, new, 1)
    compile(source, "launch.py", "exec")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch.py").write_text(source, encoding="utf-8")
    shutil.copyfile(folder / "kernel-metadata.json", output / "kernel-metadata.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "bundle")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args()
    if any(value is not None and value <= 0 for value in (args.epochs, args.max_train_samples, args.max_val_samples)):
        parser.error("epochs and sample limits must be positive")
    print(build(**vars(args)))
