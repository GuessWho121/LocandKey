# Generalization Improvement Checklist

- [x] Freeze the Phoenix holdout result; do not tune against it again.
- [x] Change the network to zero-initialized residual correction.
- [x] Condition correction strength on the requested SNR.
- [x] Add deterministic global-phase augmentation and broader complex noise.
- [x] Add an identity safety fallback for corrections that fail a noise-consistency check.
- [x] Use San Francisco as the development holdout with matched baseline evaluation.
- [x] Add focused regression checks and pass the existing local test suite.
- [x] Select and document one new downloadable CSI scenario.
- [x] Pass a two-GPU Kaggle smoke run.
- [x] Submit full Kaggle DDP training and monitor it quietly in the background.

Phoenix remains report-only evidence. Model selection and implementation decisions
must use the San Francisco development holdout or fixed leave-one-scenario-out
rules, not repeated Phoenix results.

## New External Scenario

Use `city_17_seattle_28` from the official DeepMIMO database. It is another
28 GHz urban environment, so it tests transfer to a new city without changing
the carrier band at the same time.

```powershell
.genvenv\Scripts\python.exe dgen_o1_28.py --scenario city_17_seattle_28 --max-samples 20000
```

The generator downloads the scenario through DeepMIMO v4 and writes
`data/generated/city_17_seattle_28/channels.npy` in the existing
`(N, 128, 256, 2)` float32 format. The official scenario contains 5,633 usable
receiver samples, so the requested maximum of 20,000 produced all 5,633. Keep
Seattle outside model selection and use it only after the San Francisco
development experiment is fixed.
