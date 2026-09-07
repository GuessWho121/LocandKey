"""
DeepMIMO Multi-Scenario Channel Generator
=========================================
Generates CSI tensors in the Resonance format:

    (N, 128, 256, 2) float32

Preferred path: DeepMIMO v4 API.
Fallback path : legacy O1_28 .mat files already supported by this project.

Examples:
    python dgen_o1_28.py --dry-run
    python dgen_o1_28.py --scenario asu_campus_3p5 --max-samples 10
"""

import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SCENARIOS = [
    "o1_28",
    "city_4_phoenix_28",
    "city_16_sanfrancisco_28",
    "asu_campus_3p5",
    "i1_2p4",
]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("LOCANDKEY_DATA_ROOT", os.path.join(PROJECT_DIR, "data"))
DEEPMIMO_ROOT = os.environ.get("LOCANDKEY_DEEPMIMO_ROOT", os.path.join(DATA_ROOT, "deepmimo"))
OUTPUT_ROOT = os.environ.get("LOCANDKEY_OUTPUT_ROOT", os.path.join(DATA_ROOT, "generated"))

LEGACY_O1_FOLDER = os.environ.get("LOCANDKEY_LEGACY_O1_FOLDER", os.path.join(DATA_ROOT, "o1_28"))

BS_IDX = 3
TX_ANT_IDX = 0

N_ANT_H = 16
N_ANT_V = 8
N_ANT = N_ANT_H * N_ANT_V
ANT_SPACING = 0.5

N_SUBCARRIERS = 256
BANDWIDTH = 50e6
NUM_PATHS = 25
DOPPLER = False

MAX_ROWS = 5
MAX_USERS_PER_ROW = 5000


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def to_iq(H):
    return np.stack([np.real(H), np.imag(H)], axis=-1).astype(np.float32)


def scenario_output_dir(output_root, scenario):
    return os.path.join(output_root, scenario)


def channels_path(output_root, scenario):
    return os.path.join(scenario_output_dir(output_root, scenario), "channels.npy")


def metadata_for(scenario, method, iq, source, max_samples, **extra):
    metadata = {
        "scenario": scenario,
        "method": method,
        "source": source,
        "shape": list(iq.shape),
        "dtype": str(iq.dtype),
        "max_samples": max_samples,
        "bs_antenna_shape": [N_ANT_H, N_ANT_V],
        "ue_antenna_shape": [1, 1],
        "antenna_spacing_wavelengths": ANT_SPACING,
        "subcarriers": N_SUBCARRIERS,
        "bandwidth_hz": BANDWIDTH,
        "num_paths": NUM_PATHS,
        "doppler": DOPPLER,
        "format": "(samples, bs_antennas, subcarriers, iq)",
    }
    metadata.update(extra)
    return metadata


def save_channels(scenario, iq, output_root, method, source, max_samples):
    if iq.ndim != 4 or iq.shape[1:] != (N_ANT, N_SUBCARRIERS, 2):
        raise ValueError(f"Unexpected output shape {iq.shape}; expected (N, {N_ANT}, {N_SUBCARRIERS}, 2)")
    if iq.dtype != np.float32:
        raise ValueError(f"Unexpected dtype {iq.dtype}; expected float32")
    if not np.isfinite(iq).all():
        raise ValueError("Generated CSI contains NaN or Inf")

    out_dir = scenario_output_dir(output_root, scenario)
    os.makedirs(out_dir, exist_ok=True)

    path = channels_path(output_root, scenario)
    metadata_path = os.path.join(out_dir, "metadata.json")

    np.save(path, iq)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_for(scenario, method, iq, source, max_samples), f, indent=2)

    print(f"Saved channels : {path}")
    print(f"Saved metadata : {metadata_path}")
    print(f"Shape          : {iq.shape}")
    print(f"Size           : {os.path.getsize(path) / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# DEEPMIMO V4 PATH
# ---------------------------------------------------------------------------

def import_deepmimo():
    try:
        import deepmimo as dm
    except ImportError:
        return None
    return dm


def deepmimo_params(dm):
    params = dm.ChannelParameters()
    params.bs_antenna.shape = [N_ANT_H, N_ANT_V]
    params.bs_antenna.spacing = ANT_SPACING
    params.ue_antenna.shape = [1, 1]
    params.ue_antenna.spacing = ANT_SPACING
    params.freq_domain = True
    params.doppler = DOPPLER
    params.num_paths = NUM_PATHS
    params.ofdm.subcarriers = N_SUBCARRIERS
    params.ofdm.selected_subcarriers = np.arange(N_SUBCARRIERS)
    params.ofdm.bandwidth = BANDWIDTH
    return params


def trim_deepmimo_dataset(dataset, max_samples):
    if not max_samples:
        return dataset

    if not hasattr(dataset, "trim"):
        print("Warning: DeepMIMO dataset has no trim(); will slice after channel generation.")
        return dataset

    idxs = None
    if hasattr(dataset, "get_idxs"):
        try:
            active = np.asarray(dataset.get_idxs("active"))
            idxs = active[:max_samples] if len(active) else None
        except Exception:
            idxs = None

    if idxs is None and hasattr(dataset, "num_paths"):
        paths = np.asarray(dataset.num_paths)
        if paths.ndim > 1:
            paths = np.any(paths > 0, axis=tuple(range(1, paths.ndim)))
        idxs = np.where(paths > 0)[0][:max_samples]

    if idxs is None or len(idxs) == 0:
        for attr in ("n_ue", "rx_pos", "rx_locations", "power"):
            if not hasattr(dataset, attr):
                continue
            value = getattr(dataset, attr)
            count = int(value) if np.isscalar(value) else len(value)
            idxs = np.arange(min(max_samples, count))
            break

    if idxs is None or len(idxs) == 0:
        raise ValueError("No receiver samples available after trimming")
    return dataset.trim(idxs=idxs)


def deepmimo_set_ids(scenario_dir):
    with open(os.path.join(scenario_dir, "params.json"), encoding="utf-8") as f:
        sets = json.load(f)["txrx_sets"].values()

    tx_set = next((item for item in sets if item.get("is_tx")), None)
    rx_set = next((item for item in sets if item.get("is_rx") and not item.get("is_tx")), None)
    if rx_set is None:
        rx_set = next((item for item in sets if item.get("is_rx")), None)
    if tx_set is None or rx_set is None:
        raise ValueError(f"No usable TX/RX set pair found in {scenario_dir}")
    return [tx_set["id"]], [rx_set["id"]], rx_set["num_points"], tx_set["num_points"]


def load_deepmimo_dataset(dm, scenario, root_dir, max_samples, tx_idx=0, rx_indices=None):
    os.makedirs(root_dir, exist_ok=True)
    old_cwd = os.getcwd()
    try:
        os.chdir(root_dir)
        scenario_dir = os.path.join(root_dir, "deepmimo_scenarios", scenario)
        zip_path = os.path.join(root_dir, "deepmimo_scenarios", f"{scenario}_downloaded.zip")
        if not os.path.isdir(scenario_dir):
            if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
                raise RuntimeError(f"Bad partial download at {zip_path}. Delete it, then retry.")
            dm.download(scenario)
        if not os.path.isdir(scenario_dir):
            raise FileNotFoundError(
                f"Scenario folder not found after download: {scenario_dir}. "
                "The download likely failed or was rate-limited; retry later."
            )
        tx_sets, rx_sets, receiver_count, _ = deepmimo_set_ids(scenario_dir)
        print(f"Using DeepMIMO sets: TX {tx_sets[0]}[{tx_idx}] -> RX {rx_sets[0]}")
        tx_sets = {tx_sets[0]: [tx_idx]}
        if rx_indices is not None:
            rx_sets = {rx_sets[0]: [int(idx) for idx in rx_indices]}
        elif max_samples:
            rx_sets = {rx_sets[0]: list(range(min(max_samples, receiver_count)))}
        try:
            return dm.load(scenario, tx_sets=tx_sets, rx_sets=rx_sets, max_paths=NUM_PATHS)
        except TypeError:
            return dm.load(scenario, tx_sets=tx_sets, rx_sets=rx_sets)
    finally:
        os.chdir(old_cwd)


def channels_from_deepmimo(dataset, dm, max_samples, already_trimmed=False):
    if isinstance(dataset, list):
        chunks = []
        remaining = max_samples
        for i, part in enumerate(dataset, start=1):
            if remaining is not None and remaining <= 0:
                break
            print(f"TX/RX dataset {i}/{len(dataset)}")
            iq = channels_from_deepmimo(part, dm, None if already_trimmed else remaining)
            chunks.append(iq)
            if remaining is not None:
                remaining -= len(iq)
        if not chunks:
            raise ValueError("No DeepMIMO TX/RX datasets produced channels")
        iq = np.concatenate(chunks, axis=0)
        return iq[:max_samples] if max_samples else iq

    dataset = trim_deepmimo_dataset(dataset, max_samples)
    if isinstance(dataset, list):
        return channels_from_deepmimo(dataset, dm, max_samples, already_trimmed=True)

    channels = dataset.compute_channels(deepmimo_params(dm))
    if channels is None:
        channels = getattr(dataset, "channel", None)
    if channels is None:
        channels = getattr(dataset, "channels", None)
    if channels is None:
        raise ValueError("DeepMIMO did not expose generated channels")

    H = np.squeeze(np.asarray(channels))
    if H.ndim == 2:
        H = H[np.newaxis, ...]
    if H.ndim != 3:
        raise ValueError(f"Cannot convert DeepMIMO channel shape {np.asarray(channels).shape}")
    if H.shape[1:] == (N_SUBCARRIERS, N_ANT):
        H = np.transpose(H, (0, 2, 1))
    if H.shape[1:] != (N_ANT, N_SUBCARRIERS):
        raise ValueError(f"Expected channel shape (N, {N_ANT}, {N_SUBCARRIERS}); got {H.shape}")
    if max_samples:
        H = H[:max_samples]
    return to_iq(H)


def generate_deepmimo_v4(scenario, root_dir, output_root, max_samples):
    dm = import_deepmimo()
    if dm is None:
        raise ImportError("DeepMIMO v4 package is not installed. Install with: pip install deepmimo")

    scenario_dir = os.path.join(root_dir, "deepmimo_scenarios", scenario)
    if not os.path.isdir(scenario_dir):
        load_deepmimo_dataset(dm, scenario, root_dir, 1)
    _, _, receiver_count, transmitter_count = deepmimo_set_ids(scenario_dir)

    if transmitter_count == 1 or not max_samples:
        dataset = load_deepmimo_dataset(dm, scenario, root_dir, max_samples)
        iq = channels_from_deepmimo(dataset, dm, max_samples)
        save_channels(scenario, iq, output_root, "deepmimo_v4", root_dir, max_samples)
        return

    out_dir = scenario_output_dir(output_root, scenario)
    os.makedirs(out_dir, exist_ok=True)
    path = channels_path(output_root, scenario)
    partial_path = path + ".partial"
    metadata_path = os.path.join(out_dir, "metadata.json")
    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(max_samples, N_ANT, N_SUBCARRIERS, 2),
    )
    transmitter_ranges = []
    offset = 0

    try:
        selected_receivers = np.linspace(0, receiver_count - 1, max_samples, dtype=int)
        for tx_idx in range(transmitter_count):
            rx_indices = selected_receivers[tx_idx::transmitter_count]
            sample_count = len(rx_indices)
            if sample_count == 0:
                continue
            print(f"Transmitter {tx_idx + 1}/{transmitter_count}: {sample_count} samples")
            dataset = load_deepmimo_dataset(dm, scenario, root_dir, sample_count, tx_idx, rx_indices)
            chunk = channels_from_deepmimo(dataset, dm, sample_count)
            if len(chunk) != sample_count:
                raise ValueError(f"Transmitter {tx_idx} produced {len(chunk)} of {sample_count} requested samples")
            if not np.isfinite(chunk).all():
                raise ValueError(f"Transmitter {tx_idx} produced NaN or Inf")
            end = offset + sample_count
            output[offset:end] = chunk
            transmitter_ranges.append(
                {
                    "tx_idx": tx_idx,
                    "start": offset,
                    "end": end,
                    "rx_index_first": int(rx_indices[0]),
                    "rx_index_last": int(rx_indices[-1]),
                }
            )
            offset = end

        output.flush()
        del output
        output = None
        os.replace(partial_path, path)
    except Exception:
        if output is not None:
            del output
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise

    saved = np.load(path, mmap_mode="r")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata_for(
                scenario,
                "deepmimo_v4_balanced_transmitters",
                saved,
                root_dir,
                max_samples,
                transmitter_count=transmitter_count,
                receiver_sampling="evenly_spaced_round_robin",
                transmitter_ranges=transmitter_ranges,
            ),
            f,
            indent=2,
        )
    print(f"Saved channels : {path}")
    print(f"Saved metadata : {metadata_path}")
    print(f"Shape          : {saved.shape}")
    print(f"Size           : {os.path.getsize(path) / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# LEGACY O1 .MAT FALLBACK
# ---------------------------------------------------------------------------

def find_row_files(folder, bs_idx, tx_idx):
    tag = f"_t{bs_idx:03d}_tx{tx_idx:03d}_r"
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    match = [f for f in files if tag in os.path.basename(f)]

    if not match:
        raise FileNotFoundError(f"No files found with tag '{tag}' in {folder}")

    rows = {}
    for fpath in match:
        fname = os.path.basename(fpath)
        key = fname.split(tag)[0]
        row_idx = int(fname.split("_r")[-1].replace(".mat", ""))
        rows.setdefault(row_idx, {})[key] = fpath

    required = {"power", "phase", "delay", "aod_az", "aod_el"}
    return {r: v for r, v in rows.items() if required.issubset(v.keys())}


def steering_vectors(aod_az_deg, aod_el_deg, n_h, n_v, spacing):
    az = np.deg2rad(aod_az_deg)
    el = np.deg2rad(aod_el_deg)

    ih = np.arange(n_h, dtype=np.float32)
    iv = np.arange(n_v, dtype=np.float32)
    ih_g, iv_g = np.meshgrid(ih, iv, indexing="xy")
    ih_f = ih_g.flatten()
    iv_f = iv_g.flatten()

    ph = 2 * np.pi * spacing * np.sin(el)[..., None] * np.cos(az)[..., None] * ih_f
    pv = 2 * np.pi * spacing * np.sin(el)[..., None] * iv_f
    return np.exp(1j * (ph + pv)).astype(np.complex64)


def build_legacy_row(files, max_users):
    import scipy.io as sio

    def load(k):
        arr = sio.loadmat(files[k])[k].astype(np.float32)
        return arr[:max_users] if max_users else arr

    power = load("power")
    phase = load("phase")
    delay = load("delay")
    aod_az = load("aod_az")
    aod_el = load("aod_el")

    valid = ~np.isnan(power)
    power = np.where(valid, power, 0.0)
    phase = np.where(valid, phase, 0.0)
    delay = np.where(valid, delay, 0.0)
    aod_az = np.where(valid, aod_az, 0.0)
    aod_el = np.where(valid, aod_el, 0.0)

    amp = np.sqrt(np.maximum(np.where(valid, np.power(10.0, power / 10.0), 0.0), 0.0))
    coeff = (amp * np.exp(1j * phase)).astype(np.complex64)
    sv = steering_vectors(aod_az, aod_el, N_ANT_H, N_ANT_V, ANT_SPACING)

    freqs = np.arange(N_SUBCARRIERS, dtype=np.float32) * (BANDWIDTH / N_SUBCARRIERS)
    ofdm = np.exp(
        -1j * 2 * np.pi * delay[:, :, None].astype(np.float64) * freqs[None, None, :]
    ).astype(np.complex64)

    return np.einsum("upa,ups->uas", coeff[:, :, None] * sv, ofdm).astype(np.complex64)


def generate_legacy_o1(folder, output_root, max_samples):
    row_files = find_row_files(folder, BS_IDX, TX_ANT_IDX)
    selected = sorted(row_files.keys())[:MAX_ROWS]
    if not selected:
        raise ValueError(f"No complete O1 rows found in {folder}")

    remaining = max_samples
    chunks = []
    for row in selected:
        if remaining is not None and remaining <= 0:
            break
        row_limit = min(MAX_USERS_PER_ROW, remaining) if remaining else MAX_USERS_PER_ROW
        print(f"Row {row:03d}: loading up to {row_limit} users")
        H = build_legacy_row(row_files[row], row_limit)
        chunks.append(H)
        if remaining is not None:
            remaining -= len(H)

    iq = to_iq(np.concatenate(chunks, axis=0))
    if max_samples:
        iq = iq[:max_samples]
    save_channels("o1_28", iq, output_root, "legacy_mat", folder, max_samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_dry_run(args):
    dm = import_deepmimo()
    scenarios = [args.scenario] if args.scenario else SCENARIOS

    print("DeepMIMO Multi-Scenario Channel Generator")
    print(f"DeepMIMO installed : {'yes' if dm else 'no'}")
    print(f"Data root          : {DATA_ROOT}")
    print(f"DeepMIMO root      : {args.root_dir}")
    print(f"Output root        : {args.output_root}")
    print(f"Shape              : (N, {N_ANT}, {N_SUBCARRIERS}, 2)")
    print(f"Max samples        : {args.max_samples or 'all'}")
    print("")
    for scenario in scenarios:
        path = channels_path(args.output_root, scenario)
        suffix = " [exists]" if os.path.exists(path) else ""
        scenario_dir = os.path.join(args.root_dir, "deepmimo_scenarios", scenario)
        params_path = os.path.join(scenario_dir, "params.json")
        sets = ""
        if os.path.isfile(params_path):
            tx_sets, rx_sets, _, transmitter_count = deepmimo_set_ids(scenario_dir)
            sets = f" [TX {tx_sets[0]} ({transmitter_count} positions) -> RX {rx_sets[0]}]"
        print(f"{scenario:24s} -> {path}{suffix}{sets}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DeepMIMO CSI in Resonance format.")
    parser.add_argument("--scenario", choices=SCENARIOS, help="Scenario to generate.")
    parser.add_argument("--max-samples", type=int, help="Limit receiver samples for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Print config and outputs without generating.")
    parser.add_argument("--root-dir", default=DEEPMIMO_ROOT, help="DeepMIMO download/load directory.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT, help="Per-scenario output directory.")
    parser.add_argument("--legacy-o1-folder", default=LEGACY_O1_FOLDER, help="Fallback folder for old O1_28 .mat files.")
    parser.add_argument("--prefer-legacy-o1", action="store_true", help="Use old O1_28 .mat path instead of DeepMIMO v4.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing channels.npy.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        print_dry_run(args)
        return
    if not args.scenario:
        raise SystemExit("Pick one scenario with --scenario, or run --dry-run to inspect the config.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")

    print("DeepMIMO Multi-Scenario Channel Generator")
    print(f"Scenario: {args.scenario}")
    print(f"Output  : {scenario_output_dir(args.output_root, args.scenario)}")
    if os.path.exists(channels_path(args.output_root, args.scenario)) and not args.overwrite:
        print("Existing channels.npy found; skipping. Use --overwrite to regenerate.")
        return

    if args.scenario == "o1_28" and args.prefer_legacy_o1:
        generate_legacy_o1(args.legacy_o1_folder, args.output_root, args.max_samples)
        return

    try:
        generate_deepmimo_v4(args.scenario, args.root_dir, args.output_root, args.max_samples)
    except ImportError:
        if args.scenario != "o1_28":
            raise
        print("DeepMIMO unavailable; falling back to legacy O1_28 .mat generation.")
        generate_legacy_o1(args.legacy_o1_folder, args.output_root, args.max_samples)


if __name__ == "__main__":
    main()
