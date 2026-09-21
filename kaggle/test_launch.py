"""Offline wiring check; no downloads or GPU use."""
from pathlib import Path
import tempfile
import runpy
from unittest.mock import patch
from build import build
import launch

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    dataset = root / "input"
    (dataset / "example").mkdir(parents=True)
    (dataset / "example" / "channels.npy").touch()
    bundled = runpy.run_path(str(build(root / "bundle") / "launch.py"))
    modules = bundled["MODULE_SOURCES"]
    for name, source in modules.items():
        assert source == (Path(__file__).resolve().parent.parent / name).read_text(encoding="utf-8")

    with patch.object(launch, "DATASET", dataset), patch.object(launch, "WORK", root / "work"), patch.object(launch, "run") as run, patch.object(launch, "MODULE_SOURCES", modules), patch.object(launch, "cuda_device_count", return_value=2), patch.object(Path, "symlink_to") as link:
        launch.main()
        link.assert_called_once_with(dataset, target_is_directory=True)
        train = run.call_args.args
        assert train[:5] == ("-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "train.py")
        assert "--require-gpu" in train
        assert train[train.index("--epochs") + 1] == "60"
        assert train[train.index("--batch-size") + 1] == "8"
        assert "_smoke" not in str(train[train.index("--weights-path") + 1])
        patched = (root / "work" / "train.py").read_text(encoding="utf-8")
        assert patched == modules["train.py"]
        assert "torch.nn.parallel.DistributedDataParallel" in patched
        assert "run_epoch(train_model," in patched
        assert "mininterval=30" in patched
        assert "refresh=False" in patched
        assert "def run_epoch(model," in patched
        assert "def run_epoch(train_model," not in patched
        compile(patched, "train.py", "exec")
        with patch.object(launch, "cuda_device_count", return_value=1):
            launch.main()
            assert run.call_args.args[0] == "train.py"
        with patch.object(launch, "SMOKE", True):
            launch.main()
            train = run.call_args.args
            assert train[train.index("--epochs") + 1] == "1"
            assert "--max-train-samples" in train
            assert "_smoke" in str(train[train.index("--weights-path") + 1])
        with patch.object(launch, "HOLDOUT", "city_4_phoenix_28"):
            try:
                launch.main()
            except ValueError:
                pass
            else:
                raise AssertionError("Holdout accepted missing original splits")
        input_root = root / "kaggle-input"
        missing = root / "missing"
        mounted = input_root / "mounted-dataset"
        (mounted / "scenario" / "channels.npy").parent.mkdir(parents=True)
        (mounted / "scenario" / "channels.npy").touch()
        with patch.object(launch, "DATASET", missing):
            assert launch.dataset_root(input_root) == mounted
        nested = input_root / "nested-dataset"
        (nested / "data" / "generated" / "scenario" / "channels.npy").parent.mkdir(parents=True)
        (nested / "data" / "generated" / "scenario" / "channels.npy").touch()
        with patch.object(launch, "DATASET", nested):
            assert launch.dataset_root(input_root) == nested / "data" / "generated"
print("Kaggle launcher checks passed")
