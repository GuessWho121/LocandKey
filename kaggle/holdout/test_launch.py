"""Verify the real baseline snapshot and reject a mismatched fingerprint."""
from pathlib import Path
import runpy
import tempfile
from unittest.mock import patch
from build import build
import launch

source = Path(__file__).resolve().parents[2] / "kaggle-output/baseline/training/locandkey"
launch.verify_source(source)
with patch.object(launch, "SPLIT_SHA256", "0" * 64):
    try:
        launch.verify_source(source)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched splits accepted")
compile(Path(launch.__file__).read_text(), "launch.py", "exec")
with tempfile.TemporaryDirectory() as folder:
    bundle = build(Path(folder), "city_16_sanfrancisco_28", smoke=True)
    values = runpy.run_path(str(bundle / "launch.py"))
    assert values["SCENARIO"] == "city_16_sanfrancisco_28" and values["SMOKE"] is True
    assert set(values["MODULE_SOURCES"]) == {"model.py", "preprocess.py", "train.py", "eval.py"}
print("PASS: baseline provenance and bundled development holdout")
