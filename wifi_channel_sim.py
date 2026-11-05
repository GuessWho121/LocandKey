"""Physically consistent paired Wi-Fi IoT CSI channel simulator for LocandKey."""

import dataclasses
import math
from typing import List, Optional, Tuple
import numpy as np


# 802.11n 20 MHz subcarrier indices: -32 to +31 (64 subcarriers).
# Standard masks: DC (index 0, offset 32), guard carriers (-32..-29 and 29..31), pilots (-21, -7, 7, 21).
SUBCARRIER_COUNT = 64
DC_INDEX = 32
GUARD_INDICES = {0, 1, 2, 3, 61, 62, 63}
PILOT_INDICES = {32 - 21, 32 - 7, 32 + 7, 32 + 21}
NULL_MASK = np.zeros(SUBCARRIER_COUNT, dtype=bool)
NULL_MASK[list(GUARD_INDICES | {DC_INDEX})] = True


@dataclasses.dataclass(frozen=True)
class MultipathProfile:
    delays_ns: np.ndarray  # (P,) in nanoseconds
    powers_linear: np.ndarray  # (P,) relative power
    angles_rad: np.ndarray  # (P,) angle of arrival


@dataclasses.dataclass(frozen=True)
class SimConfig:
    bandwidth_hz: float = 20e6
    subcarriers: int = 64
    center_freq_hz: float = 2.412e9  # Channel 1 (2.412 GHz)
    num_paths: int = 8
    delay_spread_ns: float = 50.0
    rician_k_db: Optional[float] = None  # None for Rayleigh, or e.g. 5.0 dB
    max_doppler_hz: float = 0.0  # 0.0 = static


@dataclasses.dataclass(frozen=True)
class ReceiverImpairment:
    gain: float = 1.0  # Amplitude scale [0.80, 1.15]
    slope: float = 0.0  # Frequency slope [-0.02, 0.02]
    phase_offset: float = 0.0  # Constant phase rotation [rad]
    phase_slope: float = 0.0  # Phase slope across subcarriers [rad]
    phase_drift_rate: float = 0.0  # Phase drift per second [rad/s]
    adc_bits: Optional[int] = None  # ADC quantization bits (e.g. 8 or 10), None = float32


def generate_multipath_profile(config: SimConfig, rng: np.random.Generator) -> MultipathProfile:
    """Generate physical multipath cluster parameters."""
    p = config.num_paths
    if p <= 0:
        raise ValueError("num_paths must be positive")
    delays = np.sort(rng.exponential(scale=max(config.delay_spread_ns, 1e-3), size=p))
    delays[0] = 0.0  # First arrival at t=0
    powers = np.exp(-delays / max(config.delay_spread_ns, 1e-3))
    if config.rician_k_db is not None:
        k_lin = 10.0 ** (config.rician_k_db / 10.0)
        powers[0] += k_lin * np.sum(powers[1:]) if p > 1 else k_lin
    powers /= np.sum(powers)
    angles = rng.uniform(0, 2 * math.pi, size=p)
    return MultipathProfile(delays, powers, angles)


def evaluate_underlying_channel(
    profile: MultipathProfile,
    config: SimConfig,
    timestamps_sec: np.ndarray,
    initial_phases: np.ndarray,
) -> np.ndarray:
    """Compute the true continuous channel H(t, f) across timestamps and subcarrier frequencies.
    
    Output shape: (T, num_subcarriers) complex64
    """
    times = np.asarray(timestamps_sec, dtype=np.float64)  # (T,)
    subcarrier_spacing = config.bandwidth_hz / config.subcarriers
    freq_offsets = (np.arange(config.subcarriers) - config.subcarriers // 2) * subcarrier_spacing  # (F,)

    # Doppler phase for each path p: 2 * pi * f_D * cos(theta_p) * t
    doppler_freqs = config.max_doppler_hz * np.cos(profile.angles_rad)  # (P,)
    # Shape: (T, P)
    path_phases_t = initial_phases[None, :] + 2.0 * math.pi * times[:, None] * doppler_freqs[None, :]
    path_coeffs_t = (np.sqrt(profile.powers_linear)[None, :] * np.exp(1j * path_phases_t)).astype(np.complex64)

    # Frequency response: sum_p a_p(t) * exp(-j 2 pi f tau_p)
    # tau_p in seconds: profile.delays_ns * 1e-9
    delays_sec = profile.delays_ns * 1e-9  # (P,)
    # freq_delays: (P, F)
    freq_phase = -2.0 * math.pi * (delays_sec[:, None] * freq_offsets[None, :])
    ofdm_steering = np.exp(1j * freq_phase).astype(np.complex64)  # (P, F)

    # H: (T, F) = (T, P) @ (P, F)
    return np.matmul(path_coeffs_t, ofdm_steering)


def apply_receiver_impairment(
    h_complex: np.ndarray,
    timestamps_sec: np.ndarray,
    impairment: ReceiverImpairment,
    mask_nulls: bool = True,
) -> np.ndarray:
    """Apply receiver RF/hardware distortion (gain, slope, phase offset/drift, ADC quantization)."""
    t = np.asarray(timestamps_sec, dtype=np.float64)
    h = h_complex.copy()
    num_subcarriers = h.shape[-1]
    f_norm = np.linspace(-1.0, 1.0, num_subcarriers, dtype=np.float32)

    # Amplitude gain & slope
    amplitude_filter = impairment.gain * (1.0 + impairment.slope * f_norm)
    h = h * amplitude_filter[None, :]

    # Phase rotation: constant + subcarrier slope + time drift
    phase = (
        impairment.phase_offset
        + impairment.phase_slope * f_norm[None, :]
        + impairment.phase_drift_rate * t[:, None]
    )
    h = h * np.exp(1j * phase).astype(np.complex64)

    # ADC Quantization (optional)
    if impairment.adc_bits is not None and impairment.adc_bits > 0:
        max_val = max(float(np.max(np.abs(h))), 1e-6)
        levels = 2 ** (impairment.adc_bits - 1) - 1
        real_q = np.clip(np.round(np.real(h) / max_val * levels), -levels, levels) / levels * max_val
        imag_q = np.clip(np.round(np.imag(h) / max_val * levels), -levels, levels) / levels * max_val
        h = (real_q + 1j * imag_q).astype(np.complex64)

    if mask_nulls:
        h[:, NULL_MASK] = 0.0 + 0.0j

    return h


def add_channel_noise(
    h_complex: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    interference_power: float = 0.0,
) -> np.ndarray:
    """Add complex Gaussian AWGN (and optional interference) scaled to target SNR in dB."""
    signal_power = float(np.mean(np.abs(h_complex) ** 2))
    if signal_power < 1e-12:
        return h_complex
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    std = math.sqrt(noise_power / 2.0)
    noise = (rng.normal(0, std, h_complex.shape) + 1j * rng.normal(0, std, h_complex.shape)).astype(np.complex64)

    if interference_power > 0.0:
        int_std = math.sqrt((signal_power * interference_power) / 2.0)
        # Narrowband interference on 2 random subcarriers
        int_carriers = rng.choice(h_complex.shape[-1], size=min(2, h_complex.shape[-1]), replace=False)
        int_noise = (rng.normal(0, int_std, (h_complex.shape[0], len(int_carriers)))
                     + 1j * rng.normal(0, int_std, (h_complex.shape[0], len(int_carriers))))
        noise[:, int_carriers] += int_noise.astype(np.complex64)

    return (h_complex + noise).astype(np.complex64)


def to_iq_tensor(h_complex: np.ndarray) -> np.ndarray:
    """Convert (..., subcarriers) complex to (..., subcarriers, 2) float32 [Real, Imag]."""
    return np.stack([np.real(h_complex), np.imag(h_complex)], axis=-1).astype(np.float32)


def simulate_paired_iot_session(
    config: SimConfig,
    packets_per_round: int = 8,
    num_rounds: int = 4,
    reciprocity_delay_sec: float = 0.002,  # 2 ms between Alice packet and Bob response
    packet_interval_sec: float = 0.010,  # 10 ms between packets in a round
    round_interval_sec: float = 0.100,  # 100 ms between rounds
    snr_db: float = 10.0,
    alice_impairment: Optional[ReceiverImpairment] = None,
    bob_impairment: Optional[ReceiverImpairment] = None,
    eve_correlations: Tuple[float, ...] = (0.0, 0.5, 0.9),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Simulate a complete paired physical Wi-Fi IoT channel session.
    
    Returns:
        alice_iq: (num_rounds, packets_per_round, subcarriers, 2) float32
        bob_iq:   (num_rounds, packets_per_round, subcarriers, 2) float32
        eves_iq:  List of (num_rounds, packets_per_round, subcarriers, 2) for each Eve correlation
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total_packets = num_rounds * packets_per_round
    # Timestamps for Alice and Bob
    t_alice = []
    t_bob = []
    current_time = 0.0
    for r in range(num_rounds):
        for p in range(packets_per_round):
            t_alice.append(current_time)
            t_bob.append(current_time + reciprocity_delay_sec)
            current_time += packet_interval_sec
        current_time += round_interval_sec
    t_alice = np.array(t_alice, dtype=np.float64)
    t_bob = np.array(t_bob, dtype=np.float64)

    # 1. Underlying physical propagation process
    profile = generate_multipath_profile(config, rng)
    initial_phases = rng.uniform(0, 2 * math.pi, size=config.num_paths)

    h_alice_true = evaluate_underlying_channel(profile, config, t_alice, initial_phases)
    h_bob_true = evaluate_underlying_channel(profile, config, t_bob, initial_phases)

    # 2. Hardware distortions
    imp_a = alice_impairment or ReceiverImpairment()
    imp_b = bob_impairment or ReceiverImpairment()
    h_alice_obs = apply_receiver_impairment(h_alice_true, t_alice, imp_a)
    h_bob_obs = apply_receiver_impairment(h_bob_true, t_bob, imp_b)

    # 3. Channel noise
    h_alice_noisy = add_channel_noise(h_alice_obs, snr_db, rng)
    h_bob_noisy = add_channel_noise(h_bob_obs, snr_db, rng)

    # 4. Eve observations (correlated channel + independent receiver noise)
    eves_noisy = []
    for rho in eve_correlations:
        eve_profile = generate_multipath_profile(config, rng)
        eve_initial_phases = rng.uniform(0, 2 * math.pi, size=config.num_paths)
        h_eve_indep = evaluate_underlying_channel(eve_profile, config, t_alice, eve_initial_phases)
        h_eve_clean = rho * h_alice_true + math.sqrt(max(1.0 - rho ** 2, 0.0)) * h_eve_indep
        eve_imp = ReceiverImpairment()
        h_eve_obs = apply_receiver_impairment(h_eve_clean, t_alice, eve_imp)
        eves_noisy.append(add_channel_noise(h_eve_obs, snr_db, rng))

    shape = (num_rounds, packets_per_round, config.subcarriers, 2)
    alice_iq = to_iq_tensor(h_alice_noisy).reshape(shape)
    bob_iq = to_iq_tensor(h_bob_noisy).reshape(shape)
    eves_iq = [to_iq_tensor(h).reshape(shape) for h in eves_noisy]

    return alice_iq, bob_iq, eves_iq
