"""Validation-only gate-aware checkpoint selector for LocandKey."""

from typing import Dict, List, Optional, Tuple


def evaluate_checkpoint_eligibility(
    epoch_metrics: Dict[float, Dict[str, float]],
    baseline_metrics: Optional[Dict[float, Dict[str, float]]] = None,
    min_gain_0db: float = 2.0,
    min_gain_10db: float = 1.5,
    max_retention_degradation_db: float = 1.0,
) -> Tuple[bool, List[str], float]:
    """Check if an epoch's validation metrics pass the safety and retention gates.
    
    epoch_metrics: {snr_db: {"noisy_nmse_db": float, "model_nmse_db": float, "nmse_gain_db": float}}
    baseline_metrics: optional {snr_db: {"model_nmse_db": float}} for seen-scenario retention check.

    Returns:
        eligible: bool
        rejection_reasons: List[str]
        worst_group_gain: float (minimum gain across all tested SNRs)
    """
    reasons = []
    gains = []

    for snr_db, metrics in sorted(epoch_metrics.items()):
        gain = metrics.get("nmse_gain_db", metrics["noisy_nmse_db"] - metrics["model_nmse_db"])
        gains.append(gain)
        if gain <= 0.0:
            reasons.append(f"Non-positive gain at {snr_db:g} dB: {gain:.2f} dB")

    worst_gain = min(gains) if gains else -float("inf")

    # Specific 0 dB and 10 dB minimum gain gates
    if 0.0 in epoch_metrics:
        gain_0 = epoch_metrics[0.0].get("nmse_gain_db",
                                        epoch_metrics[0.0]["noisy_nmse_db"] - epoch_metrics[0.0]["model_nmse_db"])
        if gain_0 < min_gain_0db:
            reasons.append(f"Gain at 0 dB below {min_gain_0db:.1f} dB threshold: {gain_0:.2f} dB")

    if 10.0 in epoch_metrics:
        gain_10 = epoch_metrics[10.0].get("nmse_gain_db",
                                          epoch_metrics[10.0]["noisy_nmse_db"] - epoch_metrics[10.0]["model_nmse_db"])
        if gain_10 < min_gain_10db:
            reasons.append(f"Gain at 10 dB below {min_gain_10db:.1f} dB threshold: {gain_10:.2f} dB")

    # Matched-seen baseline retention check (model NMSE should not be > 1 dB worse than frozen baseline)
    if baseline_metrics is not None:
        for snr_db, base in baseline_metrics.items():
            if snr_db in epoch_metrics:
                degradation = epoch_metrics[snr_db]["model_nmse_db"] - base["model_nmse_db"]
                if degradation > max_retention_degradation_db:
                    reasons.append(
                        f"Degradation at {snr_db:g} dB ({degradation:.2f} dB) exceeds "
                        f"retention limit ({max_retention_degradation_db:.1f} dB)"
                    )

    eligible = len(reasons) == 0
    return eligible, reasons, worst_gain


def select_best_checkpoint(
    candidate_epochs: List[Dict],
    baseline_metrics: Optional[Dict[float, Dict[str, float]]] = None,
) -> Tuple[str, Optional[Dict]]:
    """Select the best eligible checkpoint based on worst-group gain (with validation loss as tie-breaker).
    
    candidate_epochs is a list of dicts:
    [
        {
            "epoch": int,
            "val_loss": float,
            "metrics": {snr_db: {"noisy_nmse_db": float, "model_nmse_db": float, "nmse_gain_db": float}}
        },
        ...
    ]
    
    Returns:
        status: "selected" or "no_safe_candidate"
        best_candidate: dict or None
    """
    eligible_candidates = []

    for cand in candidate_epochs:
        eligible, reasons, worst_gain = evaluate_checkpoint_eligibility(
            cand["metrics"], baseline_metrics=baseline_metrics
        )
        if eligible:
            eligible_candidates.append({**cand, "worst_gain": worst_gain})

    if not eligible_candidates:
        return "no_safe_candidate", None

    # Rank primarily by highest worst-group gain, tie-break by lowest validation loss
    best = max(eligible_candidates, key=lambda c: (c["worst_gain"], -c.get("val_loss", 0.0)))
    return "selected", best
