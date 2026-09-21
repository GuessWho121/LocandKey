# Kaggle Script Training

No notebook conversion is needed. Build the private Script kernel from the four
local project modules with `build.py`. This includes local training fixes without
requiring a GitHub push. Module hashes are saved in `kaggle_run.json`.
Internet access and GPU availability must be enabled on your Kaggle account.

From the LocandKey project directory, using your authenticated Kaggle CLI:

```powershell
python .\kaggle\build.py
uv tool run --from kaggle kaggle kernels push -p .\kaggle\bundle --accelerator NvidiaTeslaT4 --timeout 43200
uv tool run --from kaggle kaggle kernels status akshatsinhabai1443/locandkey-training
uv tool run --from kaggle kaggle kernels output akshatsinhabai1443/locandkey-training -p .\kaggle-output
```

Push STARTS the run. Use the T4 command for Kaggle's two-GPU runtime. The launcher
uses `torchrun` and `DistributedDataParallel` with one persistent process per GPU,
keeps normal eval-compatible weights, and uses global batch size 8 on two GPUs.
Each batch is partitioned between ranks; uneven batches are weighted by sample
count, and empty ranks contribute zero gradients. Only rank zero writes logs and
checkpoints. The default build uses full data and 60 epochs. Use `build.py --smoke`
for 64 training and 16 validation samples, or `build.py --epochs 1` for a full-data
verification epoch. The launcher keeps the runtime's CUDA PyTorch; it does
not install the Windows/RTX 5070 wheel or the pinned laptop requirements.

CNS proj is attached automatically. If Kaggle uses a different mount path,
change DATASET to the directory directly containing the five scenario folders.
Arrays remain read-only in /kaggle/input; a symlink supplies the existing loader's
generated directory without copying 24.5 GB. Splits, weights and logs are written
under /kaggle/working/locandkey. Validation scans all channels even for a smoke run.

The uploaded dataset has no original split files. Default runs create NEW baseline
splits; do not assume these match previous laptop experiments. For Phoenix holdout,
attach the original split_summary.json and split_indices.npz in a separate private
dataset, add its owner/slug to dataset_sources in kernel-metadata.json, set
SOURCE_SPLITS to its mounted folder, and set HOLDOUT="city_4_phoenix_28".
The existing preprocessing code preserves the original indices.

For resume, retain the exact split files and full .last.pt checkpoint from the
previous run, attach them as inputs, set SOURCE_SPLITS and RESUME, and set SMOKE=False.
For holdout, retain the original source splits too and keep HOLDOUT unchanged.
The launcher checks the exact split fingerprint before resuming. Full checkpoints
restore the original schedule, so a one-epoch smoke checkpoint is not a 60-epoch
training starting point. Keep downloaded checkpoints private/trusted.

Kaggle session limits can interrupt long training. Only completed epochs have
resume checkpoints; do not assume timeout or cancellation will preserve outputs.
Download saved outputs after runs, and retain splits together with checkpoints.
This launcher trains only; eval.py is downloaded for subsequent evaluation.

Offline check (no Kaggle/GPU/network required):

```powershell
python .\kaggle\test_launch.py
python .\kaggle\test_distributed.py
```

The September 2026 Kaggle diagnostic reproduced DataParallel host RAM growth
from about 2.4 GiB to 10.1 GiB in 120 synthetic batches, at about 3.9 seconds per
batch after warm-up. DDP completed 600 synthetic batches in 34 seconds including
startup, with roughly 2 GiB RAM per process and 0.031 seconds per warmed-up batch.
These are synthetic measurements; full-data throughput includes disk and noise
generation. Raw results are in the private `locandkey-diagnostic` kernel outputs.

Metadata reference: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
