"""Dataset generator and PyTorch Sequence for Native Wi-Fi IoT CSI Simulation."""

import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from wifi_channel_sim import (
    SimConfig,
    ReceiverImpairment,
    generate_multipath_profile,
    evaluate_underlying_channel,
    apply_receiver_impairment,
    add_channel_noise,
    to_iq_tensor,
)


@dataclasses.dataclass(frozen=True)
class WiFiScenarioSpec:
    name: str
    delay_spread_ns: float
    rician_k_db: Optional[float]
    max_doppler_hz: float
    sample_count: int


# Five representative Wi-Fi environments (Phase B2)
WIFI_SCENARIOS = [
    WiFiScenarioSpec("small_room", delay_spread_ns=15.0, rician_k_db=10.0, max_doppler_hz=0.0, sample_count=1000),
    WiFiScenarioSpec("open_indoor", delay_spread_ns=30.0, rician_k_db=5.0, max_doppler_hz=2.0, sample_count=1000),
    WiFiScenarioSpec("corridor", delay_spread_ns=50.0, rician_k_db=0.0, max_doppler_hz=5.0, sample_count=1000),
    WiFiScenarioSpec("obstructed_indoor", delay_spread_ns=100.0, rician_k_db=None, max_doppler_hz=10.0, sample_count=1000),
    WiFiScenarioSpec("outdoor_short_range", delay_spread_ns=200.0, rician_k_db=3.0, max_doppler_hz=20.0, sample_count=1000),
]


class WiFiIoTDataset:
    """In-memory or disk-backed dataset of paired/single Wi-Fi IoT CSI measurements."""

    def __init__(
        self,
        samples: np.ndarray,
        snr_db: np.ndarray,
        clean_samples: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        self.samples = samples  # (N, subcarriers, 2)
        self.snr_db = snr_db  # (N,)
        self.clean_samples = clean_samples  # (N, subcarriers, 2)
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, float]:
        return self.samples[index], self.clean_samples[index], float(self.snr_db[index])


def generate_wifi_iot_split(
    scenarios: List[WiFiScenarioSpec] = WIFI_SCENARIOS,
    snr_grid: Tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
    subcarriers: int = 64,
    seed: int = 42,
    clean_fraction: float = 0.50,
    single_impairment_fraction: float = 0.25,
) -> Tuple[WiFiIoTDataset, WiFiIoTDataset, WiFiIoTDataset]:
    """Generate deterministic train (70%), val (15%), test (15%) Wi-Fi IoT datasets.
    
    Impairment mixture (Phase C2):
    - 50% clean + AWGN
    - 25% single receiver impairment
    - 25% combined impairments
    """
    rng = np.random.default_rng(seed)
    train_noisy, train_clean, train_snr = [], [], []
    val_noisy, val_clean, val_snr = [], [], []
    test_noisy, test_clean, test_snr = [], [], []

    for spec in scenarios:
        config = SimConfig(
            subcarriers=subcarriers,
            delay_spread_ns=spec.delay_spread_ns,
            rician_k_db=spec.rician_k_db,
            max_doppler_hz=spec.max_doppler_hz,
        )

        n = spec.sample_count
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val

        # Generate continuous channel samples across time
        timestamps = np.linspace(0.0, n * 0.010, n, dtype=np.float64)
        profile = generate_multipath_profile(config, rng)
        phases = rng.uniform(0, 2 * math.pi, size=config.num_paths)
        h_true = evaluate_underlying_channel(profile, config, timestamps, phases)

        # Normalize clean channel to unit power
        norm_factor = np.sqrt(np.mean(np.abs(h_true) ** 2, axis=1, keepdims=True) + 1e-12)
        h_clean = (h_true / norm_factor).astype(np.complex64)

        for i in range(n):
            snr = float(rng.choice(snr_grid))
            mode_draw = rng.uniform(0.0, 1.0)
            if mode_draw < clean_fraction:
                # Noise-only
                imp = ReceiverImpairment()
            elif mode_draw < clean_fraction + single_impairment_fraction:
                # Single impairment (gain variation only)
                imp = ReceiverImpairment(gain=float(rng.uniform(0.80, 1.15)))
            else:
                # Combined impairments (gain + slope + phase drift)
                imp = ReceiverImpairment(
                    gain=float(rng.uniform(0.80, 1.15)),
                    slope=float(rng.uniform(-0.02, 0.02)),
                    phase_offset=float(rng.uniform(-math.pi, math.pi)),
                    phase_slope=float(rng.uniform(-0.1, 0.1)),
                )

            h_distorted = apply_receiver_impairment(h_clean[i:i+1], timestamps[i:i+1], imp)
            h_noisy = add_channel_noise(h_distorted, snr, rng)

            sample_noisy = to_iq_tensor(h_noisy[0])
            sample_clean = to_iq_tensor(h_clean[i])

            if i < n_train:
                train_noisy.append(sample_noisy)
                train_clean.append(sample_clean)
                train_snr.append(snr)
            elif i < n_train + n_val:
                val_noisy.append(sample_noisy)
                val_clean.append(sample_clean)
                val_snr.append(snr)
            else:
                test_noisy.append(sample_noisy)
                test_clean.append(sample_clean)
                test_snr.append(snr)

    train_ds = WiFiIoTDataset(np.array(train_noisy), np.array(train_snr), np.array(train_clean), {"split": "train"})
    val_ds = WiFiIoTDataset(np.array(val_noisy), np.array(val_snr), np.array(val_clean), {"split": "validation"})
    test_ds = WiFiIoTDataset(np.array(test_noisy), np.array(test_snr), np.array(test_clean), {"split": "test"})
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Smoke verification
    smoke_scenarios = [WiFiScenarioSpec("smoke_room", 20.0, 5.0, 0.0, 20)]
    tr, va, te = generate_wifi_iot_split(smoke_scenarios)
    assert len(tr) == 14 and len(va) == 3 and len(te) == 3
    x, y, s = tr[0]
    assert x.shape == (64, 2) and y.shape == (64, 2)
    print(f"WiFi IoT dataset split generation verified: train={len(tr)}, val={len(va)}, test={len(te)}")
