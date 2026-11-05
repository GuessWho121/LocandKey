"""Tests for validation-only gate-aware checkpoint selector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate_aware_selector import evaluate_checkpoint_eligibility, select_best_checkpoint


def test_rejection_on_negative_gain():
    metrics = {
        0.0: {"noisy_nmse_db": 0.0, "model_nmse_db": -5.0, "nmse_gain_db": 5.0},
        10.0: {"noisy_nmse_db": -10.0, "model_nmse_db": -12.0, "nmse_gain_db": 2.0},
        30.0: {"noisy_nmse_db": -30.0, "model_nmse_db": -28.0, "nmse_gain_db": -2.0},  # Negative gain!
    }
    eligible, reasons, worst_gain = evaluate_checkpoint_eligibility(metrics)
    assert not eligible
    assert any("Non-positive gain at 30 dB" in r for r in reasons)
    assert worst_gain == -2.0


def test_rejection_on_baseline_degradation():
    metrics = {
        0.0: {"noisy_nmse_db": 0.0, "model_nmse_db": -20.0, "nmse_gain_db": 20.0},
        10.0: {"noisy_nmse_db": -10.0, "model_nmse_db": -22.0, "nmse_gain_db": 12.0},
    }
    baseline = {
        0.0: {"model_nmse_db": -24.0},  # Candidate is 4 dB worse than baseline at 0 dB!
        10.0: {"model_nmse_db": -23.0},
    }
    eligible, reasons, _ = evaluate_checkpoint_eligibility(metrics, baseline_metrics=baseline)
    assert not eligible
    assert any("exceeds retention limit" in r for r in reasons)


def test_no_safe_candidate_emitted():
    bad_epochs = [
        {
            "epoch": 1,
            "val_loss": 0.01,
            "metrics": {
                0.0: {"noisy_nmse_db": 0.0, "model_nmse_db": -1.0, "nmse_gain_db": 1.0},  # Below 2 dB gate
                10.0: {"noisy_nmse_db": -10.0, "model_nmse_db": -11.0, "nmse_gain_db": 1.0},
            },
        }
    ]
    status, candidate = select_best_checkpoint(bad_epochs)
    assert status == "no_safe_candidate"
    assert candidate is None


def test_successful_selection():
    epochs = [
        {
            "epoch": 10,
            "val_loss": 0.005,
            "metrics": {
                0.0: {"noisy_nmse_db": 0.0, "model_nmse_db": -15.0, "nmse_gain_db": 15.0},
                10.0: {"noisy_nmse_db": -10.0, "model_nmse_db": -20.0, "nmse_gain_db": 10.0},
                30.0: {"noisy_nmse_db": -30.0, "model_nmse_db": -32.0, "nmse_gain_db": 2.0},
            },
        },
        {
            "epoch": 20,
            "val_loss": 0.003,
            "metrics": {
                0.0: {"noisy_nmse_db": 0.0, "model_nmse_db": -18.0, "nmse_gain_db": 18.0},
                10.0: {"noisy_nmse_db": -10.0, "model_nmse_db": -22.0, "nmse_gain_db": 12.0},
                30.0: {"noisy_nmse_db": -30.0, "model_nmse_db": -34.0, "nmse_gain_db": 4.0},
            },
        },
    ]
    status, best = select_best_checkpoint(epochs)
    assert status == "selected"
    assert best["epoch"] == 20
    assert best["worst_gain"] == 4.0


if __name__ == "__main__":
    test_rejection_on_negative_gain()
    test_rejection_on_baseline_degradation()
    test_no_safe_candidate_emitted()
    test_successful_selection()
    print("All gate-aware checkpoint selector tests passed!")
