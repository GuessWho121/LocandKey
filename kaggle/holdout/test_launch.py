"""Verify the real baseline snapshot and reject a mismatched fingerprint."""
from pathlib import Path
from unittest.mock import patch
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
print("PASS: verified baseline artifacts and split mismatch rejection")
