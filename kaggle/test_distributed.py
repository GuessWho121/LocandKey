"""CPU DDP check: uneven and empty rank slices preserve a single-GPU update."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from train import run_epoch


class TinyModel(torch.nn.Conv2d):
    def forward(self, x, snr_db):
        return super().forward(x)


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    rng = np.random.default_rng(42)
    data = []
    for size in (4, 3, 1):
        clean = rng.normal(size=(size, 4, 4, 2)).astype(np.float32)
        clean /= np.sqrt((clean * clean).sum(axis=(1, 2, 3), keepdims=True))
        data.append((clean + .01, clean, np.full(size, 10, dtype=np.float32)))
    baseline = TinyModel(2, 2, 1)
    model = TinyModel(2, 2, 1)
    model.load_state_dict(baseline.state_dict())
    def run(model):
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return run_epoch(model, data, torch.device("cpu"), optimizer, scheduler,
                         torch.amp.GradScaler("cuda", enabled=False))
    run(baseline)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        metrics = run(wrapped)
        assert all(np.isfinite(value) for value in metrics.values())
        for actual, expected in zip(model.parameters(), baseline.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        validation = run_epoch(wrapped, data, torch.device("cpu"))
        assert all(np.isfinite(value) for value in validation.values())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(worker, args=((Path(folder) / "rendezvous").as_uri(),), nprocs=2, join=True)
    print("PASS: DDP updates, uneven batches, empty ranks, and validation")
