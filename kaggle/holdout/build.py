"""Bundle the current model into the reusable Kaggle holdout launcher."""
import argparse
import json
from pathlib import Path


def build(output, scenario, smoke=False):
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    if not scenario.replace("_", "").isalnum():
        raise ValueError("scenario must contain only letters, numbers and underscores")
    modules = {name: (root / name).read_text(encoding="utf-8")
               for name in ("model.py", "preprocess.py", "train.py", "eval.py")}
    for name, source in modules.items():
        compile(source, name, "exec")
    source = (folder / "launch.py").read_text(encoding="utf-8")
    for old, new in {
        'SCENARIO = "city_4_phoenix_28"': f"SCENARIO = {scenario!r}",
        "MODULE_SOURCES = None": f"MODULE_SOURCES = {modules!r}",
        "SMOKE = False": f"SMOKE = {smoke!r}",
    }.items():
        if source.count(old) != 1:
            raise ValueError(f"Launcher setting changed: {old}")
        source = source.replace(old, new, 1)
    compile(source, "launch.py", "exec")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = "akshatsinhabai1443/locandkey-generalization-dev"
    metadata["title"] = "LocandKey Generalization Dev"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch.py").write_text(source, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="city_16_sanfrancisco_28")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "bundle")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(build(**vars(args)))
