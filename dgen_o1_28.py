"""
O1_28 Channel Generator
=======================
Reads raw .mat ray-tracing files and builds MIMO channel matrices.
Output: channels_o1_28.npy  shape (N, 128, 256, 2)  float32

Edit the CONFIG block, then run:
    python generate_o1_28.py
"""

import os, sys, glob
import numpy as np
import scipy.io as sio

# ══════════════════════════════════════════════════════
#  CONFIG — only edit this section
# ══════════════════════════════════════════════════════

SCENARIO_FOLDER   = r"D:\ML Models\scenarios\o1_28"
OUTPUT_NPY        = r"D:\ML Models\scenarios\channels_o1_28.npy"

BS_IDX            = 3      # the t0XX number in your filenames (t003 → 3)
TX_ANT_IDX        = 0      # almost always 0

N_ANT_H           = 16     # 16 × 8 = 128 antennas total
N_ANT_V           = 8
ANT_SPACING       = 0.5    # wavelengths

N_SUBCARRIERS     = 256
BANDWIDTH         = 50e6   # 50 MHz
FC                = 28e9   # 28 GHz

MAX_ROWS          = 5      # number of user rows to process (each has ~497k users)
MAX_USERS_PER_ROW = 5000   # users to take per row → 5 × 5000 = 25,000 total

# ══════════════════════════════════════════════════════


def find_row_files(folder, bs_idx, tx_idx):
    tag   = f"_t{bs_idx:03d}_tx{tx_idx:03d}_r"
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    match = [f for f in files if tag in os.path.basename(f)]

    if not match:
        print(f"\n❌  No files found with tag '{tag}' in {folder}")
        print(f"    Sample files: {[os.path.basename(f) for f in files[:5]]}")
        print(f"    → Check BS_IDX and TX_ANT_IDX in CONFIG")
        sys.exit(1)

    rows = {}
    for fpath in match:
        fname   = os.path.basename(fpath)
        key     = fname.split(tag)[0]
        row_idx = int(fname.split("_r")[-1].replace(".mat", ""))
        rows.setdefault(row_idx, {})[key] = fpath

    required = {'power', 'phase', 'delay', 'aod_az', 'aod_el'}
    complete = {r: v for r, v in rows.items() if required.issubset(v.keys())}
    return complete


def steering_vectors(aod_az_deg, aod_el_deg, n_h, n_v, spacing):
    az  = np.deg2rad(aod_az_deg)   # (users, paths)
    el  = np.deg2rad(aod_el_deg)

    ih  = np.arange(n_h, dtype=np.float32)
    iv  = np.arange(n_v, dtype=np.float32)
    ih_g, iv_g = np.meshgrid(ih, iv, indexing='xy')
    ih_f = ih_g.flatten()          # (n_ant,)
    iv_f = iv_g.flatten()

    sin_el = np.sin(el)            # (users, paths)
    cos_az = np.cos(az)

    ph = 2*np.pi*spacing * sin_el[...,None] * cos_az[...,None] * ih_f
    pv = 2*np.pi*spacing * sin_el[...,None] * iv_f

    return np.exp(1j*(ph+pv)).astype(np.complex64)   # (users, paths, n_ant)


def build_row(files, n_h, n_v, n_sub, bw, spacing, max_users):
    def load(k):
        arr = sio.loadmat(files[k])[k].astype(np.float32)
        return arr[:max_users] if max_users else arr

    power  = load('power')
    phase  = load('phase')
    delay  = load('delay')
    aod_az = load('aod_az')
    aod_el = load('aod_el')

    # ── FIX 1: Zero out NaN paths (missing/inactive rays) ──
    valid  = ~np.isnan(power)
    power  = np.where(valid, power,  0.0)
    phase  = np.where(valid, phase,  0.0)
    delay  = np.where(valid, delay,  0.0)
    aod_az = np.where(valid, aod_az, 0.0)
    aod_el = np.where(valid, aod_el, 0.0)

    # ── FIX 2: Convert dB → linear power ──
    # power is in dBW or dBm — convert: linear = 10^(dB/10)
    power_linear = np.power(10.0, power / 10.0)
    # Zero out paths that were NaN (they'd have gotten a value from the conversion)
    power_linear = np.where(valid, power_linear, 0.0)

    amp   = np.sqrt(np.maximum(power_linear, 0))
    coeff = (amp * np.exp(1j * phase)).astype(np.complex64)

    sv    = steering_vectors(aod_az, aod_el, n_h, n_v, spacing)

    freqs = np.arange(n_sub, dtype=np.float32) * (bw / n_sub)
    ofdm  = np.exp(-1j * 2*np.pi
                   * delay[:,:,None].astype(np.float64)
                   * freqs[None,None,:]).astype(np.complex64)

    weighted = coeff[:,:,None] * sv
    H = np.einsum('upa,ups->uas', weighted, ofdm).astype(np.complex64)
    return H                                                      # (u, a, s)


def to_iq(H):
    return np.stack([np.real(H), np.imag(H)], axis=-1).astype(np.float32)


def main():
    print("\n" + "═"*55)
    print("  O1_28 Channel Generator")
    print("═"*55)
    print(f"  Folder  : {SCENARIO_FOLDER}")
    print(f"  Output  : {OUTPUT_NPY}")
    print(f"  Config  : {N_ANT_H}×{N_ANT_V} antennas | {N_SUBCARRIERS} subcarriers")
    print(f"  Rows    : {MAX_ROWS} × {MAX_USERS_PER_ROW} users = "
        f"{MAX_ROWS*MAX_USERS_PER_ROW:,} total")

    # Find files
    print(f"\n[1/3] Scanning folder...")
    row_files   = find_row_files(SCENARIO_FOLDER, BS_IDX, TX_ANT_IDX)
    all_rows    = sorted(row_files.keys())
    selected    = all_rows[:MAX_ROWS]
    print(f"      Found {len(all_rows)} complete rows. Using: {selected}")

    # Build channels
    print(f"\n[2/3] Building channels...")
    chunks = []
    for r in selected:
        print(f"      Row {r:03d} ... ", end="", flush=True)
        H = build_row(row_files[r], N_ANT_H, N_ANT_V,
                    N_SUBCARRIERS, BANDWIDTH, ANT_SPACING, MAX_USERS_PER_ROW)
        print(f"{H.shape}  ✅")
        chunks.append(H)

    H_all = np.concatenate(chunks, axis=0)   # (N, 128, 256)
    iq    = to_iq(H_all)                     # (N, 128, 256, 2)
    print(f"\n      Final shape : {iq.shape}   dtype: {iq.dtype}")

    # Save
    print(f"\n[3/3] Saving...")
    np.save(OUTPUT_NPY, iq)
    mb = os.path.getsize(OUTPUT_NPY) / 1e6
    print(f"      ✅  {OUTPUT_NPY}")
    print(f"      Size  : {mb:.1f} MB")
    print(f"\n  Done! Next step → run preprocess_2.py on this file.")
    print("═"*55 + "\n")


if __name__ == "__main__":
    main()