"""Offline input discovery and artifact integrity check."""
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import evaluate


with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    source = root / "kernels" / "training" / "locandkey"
    for name in evaluate.EXPECTED:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    dataset = root / "datasets" / "owner" / "cns-proj"
    channel = dataset / "scenario" / "channels.npy"
    channel.parent.mkdir(parents=True)
    channel.touch()
    hashes = {name: hashlib.sha256(b"fixture").hexdigest() for name in evaluate.EXPECTED}
    with patch.object(evaluate, "EXPECTED", hashes):
        assert evaluate.evaluation_inputs(root) == (source, dataset)
        (source / "weights/multiscenario_best.pt").write_bytes(b"changed")
        try:
            evaluate.evaluation_inputs(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed weights were accepted")
print("PASS: nested mounts and mismatched weights rejected")
