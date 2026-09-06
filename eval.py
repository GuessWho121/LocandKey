"""
Resonance - Evaluation & Visualization Engine v4
==================================================
Produces:
  1. Heatmaps       — Noisy vs Clean vs AI Predicted (I and Q)
  2. Spectral Plot  — Channel impulse response comparison
  3. NMSE vs SNR    — Corrupted Input vs Resonance (Ours)
  4. BER vs SNR     — Corrupted Input vs Resonance (Ours)
  5. Training Curves— Loss + NMSE dB from CSV log
  6. Summary Poster — Single-page overview for presentation

Run:
    python eval.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, csv
import tensorflow as tf
from scipy.special import erfc

from model import build_resonance_model, nmse_metric, nmse_db_metric, resonance_loss

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
WEIGHTS_PATH = os.path.join("weights", "resonance_best.weights.h5")
SPLITS_DIR   = r"D:\ML Models\scenarios\splits"
LOGS_DIR     = "logs"
SAVE_DIR     = "visualizations"

BATCH_SIZE   = 16
SNR_RANGE_DB = np.arange(-10, 31, 2)
N_EVAL       = 200

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccee",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#ccccee",
    "ytick.color":      "#ccccee",
    "text.color":       "#ffffff",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "legend.framealpha": 0.3,
    "legend.edgecolor": "#444466",
})
C_NOISY  = "#ff6b6b"
C_AI     = "#51cf66"
C_TRUTH  = "#00d4ff"
C_PURPLE = "#7b61ff"
# ══════════════════════════════════════════════════════

os.makedirs(SAVE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def to_complex(t):
    return (t[..., 0] + 1j * t[..., 1]).astype(np.complex64)

def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2.0))

def ber_64qam(sinr):
    return (4/6) * q_func(np.sqrt(3 * np.clip(sinr, 1e-5, 1e5) / 63))

def nmse_db(H_true, H_est):
    err    = np.mean(np.abs(H_true - H_est) ** 2)
    energy = np.mean(np.abs(H_true) ** 2) + 1e-10
    return 10 * np.log10(err / energy)

def mean_ber(H_true, H_est):
    sig   = np.mean(np.abs(H_true) ** 2,          axis=(1, 2))
    noise = np.mean(np.abs(H_true - H_est) ** 2,  axis=(1, 2))
    return np.mean(ber_64qam(sig / (noise + 1e-10)))

def savefig(name):
    path = os.path.join(SAVE_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"   Saved → {path}")
    plt.close()


# ══════════════════════════════════════════════════════
#  LOAD MODEL + DATA
# ══════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  Resonance — Evaluation Engine v4")
print("═"*60)

print("\n[1/6] Loading model and weights...")
model = build_resonance_model((128, 256, 2))
model.compile(optimizer="adam", loss=resonance_loss(),
              metrics=[nmse_metric, nmse_db_metric])
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"No weights at {WEIGHTS_PATH}")
model.load_weights(WEIGHTS_PATH)
print("   ✅ Weights loaded.")

print("\n[2/6] Loading test data...")
X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
H_clean = to_complex(X_test)
print(f"   Test set: {X_test.shape}")

np.random.seed(42)
noisy_input = (X_test + np.random.normal(0, 0.1, X_test.shape)).astype(np.float32)

print("\n[3/6] Running inference...")
predictions  = model.predict(noisy_input, batch_size=BATCH_SIZE, verbose=1)
overall_nmse = nmse_db(H_clean, to_complex(predictions))
print(f"   Overall NMSE: {overall_nmse:.2f} dB")


# ══════════════════════════════════════════════════════
#  SNR SWEEP
# ══════════════════════════════════════════════════════

print(f"\n[4/6] SNR sweep ({len(SNR_RANGE_DB)} points × {N_EVAL} samples)...")

nmse_noisy_list, nmse_ai_list = [], []
ber_noisy_list,  ber_ai_list  = [], []

H_true_eval = H_clean[:N_EVAL]
X_true_eval = X_test[:N_EVAL]

for snr_db in SNR_RANGE_DB:
    sig_power = np.mean(np.abs(H_true_eval) ** 2)
    noise_std = np.sqrt(sig_power / (2 * 10 ** (snr_db / 10.0)))
    noise_iq  = np.random.normal(0, noise_std, X_true_eval.shape).astype(np.float32)
    noisy_snr = X_true_eval + noise_iq
    H_noisy   = to_complex(noisy_snr)

    pred_snr  = model.predict(noisy_snr, batch_size=BATCH_SIZE, verbose=0)
    H_ai      = to_complex(pred_snr)

    nmse_noisy_list.append(nmse_db(H_true_eval, H_noisy))
    nmse_ai_list   .append(nmse_db(H_true_eval, H_ai))
    ber_noisy_list .append(mean_ber(H_true_eval, H_noisy))
    ber_ai_list    .append(mean_ber(H_true_eval, H_ai))

    print(f"   SNR {snr_db:+3d} dB  |  "
          f"NMSE  Noisy: {nmse_noisy_list[-1]:6.1f} dB  "
          f"AI: {nmse_ai_list[-1]:6.1f} dB  |  "
          f"BER  Noisy: {ber_noisy_list[-1]:.2e}  "
          f"AI: {ber_ai_list[-1]:.2e}")

nmse_noisy_arr = np.array(nmse_noisy_list)
nmse_ai_arr    = np.array(nmse_ai_list)
ber_noisy_arr  = np.array(ber_noisy_list)
ber_ai_arr     = np.array(ber_ai_list)


# ══════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════

print("\n[5/6] Generating plots...")

# ── Heatmaps ─────────────────────────────────────────
def plot_heatmaps(sample_idx=0):
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        f"Massive MIMO Channel Estimation — Sample {sample_idx}\n"
        "128 Antennas × 256 Subcarriers  |  ConvNeXt U-Net + Attention + CBAM",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.3)
    titles   = ["Noisy Input", "Ground Truth", "Resonance", "Residual Error"]
    cmaps    = ["magma", "viridis", "viridis", "RdBu_r"]
    ch_names = ["Real (I)", "Imaginary (Q)"]

    for row in range(2):
        ng = noisy_input[sample_idx, :, :, row]
        tg = X_test[sample_idx, :, :, row]
        pg = predictions[sample_idx, :, :, row]
        eg = tg - pg
        vmin, vmax = tg.min(), tg.max()
        for col, (grid, title, cmap) in enumerate(zip([ng,tg,pg,eg], titles, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            lim = max(abs(eg.min()), abs(eg.max()))
            im  = ax.imshow(grid, aspect="auto", cmap=cmap,
                            vmin=(-lim if col==3 else vmin),
                            vmax=( lim if col==3 else vmax))
            ax.set_title(f"{title}\n{ch_names[row]}", fontsize=10)
            ax.set_xlabel("Subcarriers", fontsize=8)
            ax.set_ylabel("Antennas",    fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    savefig(f"heatmaps_sample{sample_idx}.png")

plot_heatmaps(0)
plot_heatmaps(1)

# ── Spectral ──────────────────────────────────────────
def plot_spectral(sample_idx=0, antenna_idx=0):
    t_ir = np.abs(np.fft.ifft(to_complex(X_test[sample_idx:sample_idx+1])[0, antenna_idx]))
    n_ir = np.abs(np.fft.ifft(to_complex(noisy_input[sample_idx:sample_idx+1])[0, antenna_idx]))
    p_ir = np.abs(np.fft.ifft(to_complex(predictions[sample_idx:sample_idx+1])[0, antenna_idx]))
    d    = np.arange(256)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Channel Impulse Response — Sample {sample_idx}, Antenna {antenna_idx}",
                 fontsize=13, fontweight="bold")
    for ax, sl, title in zip(axes,
                              [slice(None), slice(40)],
                              ["Full Response", "Zoom: First 40 Taps (Dominant Paths)"]):
        ax.plot(d[sl], t_ir[sl], color=C_TRUTH, lw=2.0, label="Ground Truth")
        ax.plot(d[sl], n_ir[sl], color=C_NOISY, lw=1.2, label="Noisy Input",  alpha=0.7)
        ax.plot(d[sl], p_ir[sl], color=C_AI,    lw=1.8, label="Resonance (Ours)", ls="--")
        ax.set_xlabel("Delay Tap"); ax.set_ylabel("|h(τ)|")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10); ax.grid(True)
    plt.tight_layout()
    savefig(f"spectral_sample{sample_idx}.png")

plot_spectral(0)

# ── NMSE vs SNR ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(SNR_RANGE_DB, nmse_noisy_arr, color=C_NOISY, lw=2,   marker="o", ms=5,
        label="Corrupted Input (No Processing)")
ax.plot(SNR_RANGE_DB, nmse_ai_arr,    color=C_AI,    lw=2.5, marker="s", ms=5,
        label="Resonance (Ours)", ls="--")
ax.set_xlabel("Input SNR (dB)", fontsize=13)
ax.set_ylabel("NMSE (dB)",      fontsize=13)
ax.set_title("Channel Estimation Quality: NMSE vs SNR\n"
             "128-Antenna Massive MIMO  |  O1_28 @ 28 GHz  |  64-QAM",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=12); ax.grid(True); ax.invert_yaxis()
idx_0 = np.argmin(np.abs(SNR_RANGE_DB - 0))
gain  = nmse_noisy_arr[idx_0] - nmse_ai_arr[idx_0]
ax.annotate(f"AI gain: {gain:.1f} dB\nat SNR=0 dB",
            xy=(0, nmse_ai_arr[idx_0]),
            xytext=(6, nmse_ai_arr[idx_0] - 4),
            arrowprops=dict(arrowstyle="->", color="white"),
            fontsize=11, color="white")
plt.tight_layout()
savefig("nmse_vs_snr.png")

# ── BER vs SNR ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogy(SNR_RANGE_DB, ber_noisy_arr, color=C_NOISY, lw=2,   marker="o", ms=5,
            label="Corrupted Input (No Processing)")
ax.semilogy(SNR_RANGE_DB, ber_ai_arr,    color=C_AI,    lw=2.5, marker="s", ms=5,
            label="Resonance (Ours)", ls="--")
ax.axhline(1e-3, color="#888888", lw=1.2, ls=":", alpha=0.7)
ax.text(SNR_RANGE_DB[-1]+0.3, 1e-3, "10⁻³\n(5G NR target)", fontsize=9,
        va="center", color="#888888")
ax.set_xlabel("Input SNR (dB)", fontsize=13)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=13)
ax.set_title("Link Reliability: BER vs SNR\n"
             "64-QAM Modulation  |  Massive MIMO  |  O1_28 @ 28 GHz",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=12); ax.grid(True, which="both"); ax.set_ylim(bottom=1e-6)
plt.tight_layout()
savefig("ber_vs_snr.png")

# ── Training Curves ───────────────────────────────────
csv_path = os.path.join(LOGS_DIR, "training_log.csv")
if os.path.exists(csv_path):
    log = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                log.setdefault(k, []).append(float(v) if v else np.nan)
    epochs = log.get("epoch", list(range(len(next(iter(log.values()))))))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training History — Resonance v2", fontsize=14, fontweight="bold")
    for key, label, color, ls in [
        ("loss",              "Train Loss", C_PURPLE, "-"),
        ("val_loss",          "Val Loss",   C_TRUTH,  "--"),
    ]:
        if key in log:
            axes[0].plot(epochs, log[key], color=color, lw=2, label=label, ls=ls)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Multi-Component Loss"); axes[0].legend(); axes[0].grid(True)
    for key, label, color, ls in [
        ("nmse_db_metric",     "Train NMSE", C_PURPLE, "-"),
        ("val_nmse_db_metric", "Val NMSE",   C_TRUTH,  "--"),
    ]:
        if key in log:
            axes[1].plot(epochs, log[key], color=color, lw=2, label=label, ls=ls)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("NMSE (dB)")
    axes[1].set_title("NMSE (dB) — Lower is Better"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    savefig("training_curves.png")
else:
    print(f"   ⚠️  No CSV log at {csv_path}")


# ══════════════════════════════════════════════════════
#  SUMMARY POSTER
# ══════════════════════════════════════════════════════

print("\n[6/6] Building summary poster...")

fig = plt.figure(figsize=(24, 16))
fig.suptitle(
    "Resonance  —  Deep Learning-Based Massive MIMO Channel Estimation\n"
    "ConvNeXt U-Net + Attention Gates + CBAM  |  Physics-Informed Multi-Component Loss  |  O1_28 @ 28 GHz",
    fontsize=16, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.45, wspace=0.35,
                       top=0.93, bottom=0.06, left=0.06, right=0.97)

# Row 0: heatmaps
titles_h = ["Noisy Input", "Ground Truth", "Resonance", "Residual Error"]
cmaps_h  = ["magma", "viridis", "viridis", "RdBu_r"]
tg = X_test[0, :, :, 0];  ng = noisy_input[0, :, :, 0]
pg = predictions[0, :, :, 0];  eg = tg - pg
vmin, vmax = tg.min(), tg.max()
for col, (grid, title, cmap) in enumerate(zip([ng,tg,pg,eg], titles_h, cmaps_h)):
    ax = fig.add_subplot(gs[0, col])
    lim = max(abs(eg.min()), abs(eg.max()))
    im  = ax.imshow(grid, aspect="auto", cmap=cmap,
                    vmin=(-lim if col==3 else vmin),
                    vmax=( lim if col==3 else vmax))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Subcarriers", fontsize=8); ax.set_ylabel("Antennas", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

# Row 1 left: NMSE vs SNR
ax_n = fig.add_subplot(gs[1, :2])
ax_n.plot(SNR_RANGE_DB, nmse_noisy_arr, color=C_NOISY, lw=2,   marker="o", ms=4,
          label="Corrupted Input")
ax_n.plot(SNR_RANGE_DB, nmse_ai_arr,    color=C_AI,    lw=2.5, marker="s", ms=4,
          label="Resonance (Ours)", ls="--")
ax_n.set_xlabel("SNR (dB)", fontsize=11); ax_n.set_ylabel("NMSE (dB)", fontsize=11)
ax_n.set_title("NMSE vs SNR", fontsize=12, fontweight="bold")
ax_n.legend(fontsize=10); ax_n.grid(True); ax_n.invert_yaxis()

# Row 1 right: BER vs SNR
ax_b = fig.add_subplot(gs[1, 2:])
ax_b.semilogy(SNR_RANGE_DB, ber_noisy_arr, color=C_NOISY, lw=2,   marker="o", ms=4,
              label="Corrupted Input")
ax_b.semilogy(SNR_RANGE_DB, ber_ai_arr,    color=C_AI,    lw=2.5, marker="s", ms=4,
              label="Resonance (Ours)", ls="--")
ax_b.axhline(1e-3, color="#888888", lw=1.2, ls=":", alpha=0.7)
ax_b.text(28, 1.4e-3, "5G target", fontsize=8, color="#888888")
ax_b.set_xlabel("SNR (dB)", fontsize=11); ax_b.set_ylabel("BER", fontsize=11)
ax_b.set_title("BER vs SNR  (64-QAM)", fontsize=12, fontweight="bold")
ax_b.legend(fontsize=10); ax_b.grid(True, which="both"); ax_b.set_ylim(bottom=1e-6)

# Row 2 left: Impulse response zoom
ax_s = fig.add_subplot(gs[2, :2])
t_ir = np.abs(np.fft.ifft(to_complex(X_test[0:1])[0, 0]))
n_ir = np.abs(np.fft.ifft(to_complex(noisy_input[0:1])[0, 0]))
p_ir = np.abs(np.fft.ifft(to_complex(predictions[0:1])[0, 0]))
d40  = np.arange(40)
ax_s.plot(d40, t_ir[:40], color=C_TRUTH, lw=2.0, label="Ground Truth")
ax_s.plot(d40, n_ir[:40], color=C_NOISY, lw=1.2, label="Noisy Input",  alpha=0.7)
ax_s.plot(d40, p_ir[:40], color=C_AI,    lw=1.8, label="Resonance", ls="--")
ax_s.set_xlabel("Delay Tap", fontsize=11); ax_s.set_ylabel("|h(τ)|", fontsize=11)
ax_s.set_title("Channel Impulse Response (First 40 Taps)", fontsize=12, fontweight="bold")
ax_s.legend(fontsize=10); ax_s.grid(True)

# Row 2 right: Scorecard table
idx_0  = np.argmin(np.abs(SNR_RANGE_DB - 0))
idx_10 = np.argmin(np.abs(SNR_RANGE_DB - 10))
idx_20 = np.argmin(np.abs(SNR_RANGE_DB - 20))
ber_imp = (ber_noisy_arr[idx_10] - ber_ai_arr[idx_10]) / (ber_noisy_arr[idx_10] + 1e-10) * 100

ax_t = fig.add_subplot(gs[2, 2:])
ax_t.axis("off")
rows_data = [
    ["Metric",           "Corrupted Input",  "Resonance",            "Gain"],
    ["NMSE @ SNR=0 dB",  f"{nmse_noisy_arr[idx_0]:.1f} dB",
                          f"{nmse_ai_arr[idx_0]:.1f} dB",
                          f"+{nmse_noisy_arr[idx_0]-nmse_ai_arr[idx_0]:.1f} dB ↑"],
    ["NMSE @ SNR=10 dB", f"{nmse_noisy_arr[idx_10]:.1f} dB",
                          f"{nmse_ai_arr[idx_10]:.1f} dB",
                          f"+{nmse_noisy_arr[idx_10]-nmse_ai_arr[idx_10]:.1f} dB ↑"],
    ["NMSE @ SNR=20 dB", f"{nmse_noisy_arr[idx_20]:.1f} dB",
                          f"{nmse_ai_arr[idx_20]:.1f} dB",
                          f"+{nmse_noisy_arr[idx_20]-nmse_ai_arr[idx_20]:.1f} dB ↑"],
    ["BER @ SNR=10 dB",  f"{ber_noisy_arr[idx_10]:.2e}",
                          f"{ber_ai_arr[idx_10]:.2e}",
                          f"{ber_imp:.0f}% reduction ↓"],
    ["Overall NMSE",     "—",
                          f"{overall_nmse:.2f} dB",
                          "Full test set"],
]
tbl = ax_t.table(cellText=rows_data[1:], colLabels=rows_data[0],
                  cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.90])
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a2e"); cell.set_edgecolor("#444466")
    cell.set_text_props(color="#ffffff")
    if r == 0:
        cell.set_facecolor("#2a1a5e")
        cell.set_text_props(color="#ffffff", fontweight="bold")
    if c == 3 and r > 0:
        cell.set_text_props(color="#51cf66", fontweight="bold")
ax_t.set_title("Performance Scorecard", fontsize=12, fontweight="bold", pad=12)

savefig("summary_poster.png")


# ══════════════════════════════════════════════════════
#  TERMINAL SUMMARY
# ══════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  RESONANCE — PERFORMANCE SUMMARY")
print(f"{'═'*60}")
print(f"  Scenario : O1_28 @ 28 GHz | 128×256 | 64-QAM")
print(f"  Test set : {len(X_test):,} samples")
print(f"{'─'*60}")
print(f"  {'SNR':>6}  {'Corrupted':>12}  {'AI':>10}  {'Gain':>10}")
print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}")
for idx in [idx_0, idx_10, idx_20]:
    snr  = SNR_RANGE_DB[idx]
    gain = nmse_noisy_arr[idx] - nmse_ai_arr[idx]
    print(f"  {snr:>+5} dB  "
          f"{nmse_noisy_arr[idx]:>10.1f} dB  "
          f"{nmse_ai_arr[idx]:>8.1f} dB  "
          f"{gain:>+8.1f} dB")
print(f"{'─'*60}")
print(f"  BER @ SNR=10 dB:")
print(f"    Corrupted   : {ber_noisy_arr[idx_10]:.3e}")
print(f"    Resonance : {ber_ai_arr[idx_10]:.3e}")
print(f"    Reduction   : {ber_imp:.1f}%")
print(f"  Overall NMSE  : {overall_nmse:.2f} dB")
print(f"{'═'*60}")
print(f"\n  Key file → visualizations/summary_poster.png\n")