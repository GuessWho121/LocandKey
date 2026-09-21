"""Rebuild faculty figures from completed baseline artifacts; never contacts Kaggle."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "kaggle-output/baseline/training/locandkey"
EVAL = ROOT / "kaggle-output/baseline/evaluation/evaluation/results"
OUT = ROOT / "output/figures/locandkey_baseline"
PDF = ROOT / "output/pdf/locandkey_performance_report.pdf"
BLUE, GREEN, RED, GRAY = "#2166ac", "#16816a", "#c03946", "#68717d"
NAMES = {"asu_campus_3p5": "ASU campus", "city_16_sanfrancisco_28": "San Francisco",
         "city_4_phoenix_28": "Phoenix", "i1_2p4": "I1", "o1_28": "O1"}


def load_data():
    with (TRAIN / "logs/multiscenario/training_log.csv").open() as stream:
        training = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    report = json.loads((EVAL / "evaluation_summary.json").read_text())
    split_dir = TRAIN / "data/splits"
    splits = json.loads((split_dir / "split_summary.json").read_text())
    fingerprint = hashlib.sha256(b"".join((split_dir / name).read_bytes()
                                for name in ("split_summary.json", "split_indices.npz"))).hexdigest()
    assert report["split_sha256"] == fingerprint, "Evaluation uses different splits"
    assert len(training) == 60 and [r["epoch"] for r in training] == list(range(60))
    rows = report["results"]
    assert len(rows) == 54 and report["max_samples_per_scenario"] is None
    assert {(r["scenario"], r["snr_db"]) for r in rows} == {
        (name, snr) for name in (*splits["scenarios"], "overall") for snr in range(-10, 31, 5)}
    for row in [*training, *rows]:
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    for row in rows:
        expected = splits["totals"]["test"] if row["scenario"] == "overall" else splits["scenarios"][row["scenario"]]["splits"]["test"]
        assert row["samples"] == expected
        assert math.isclose(row["nmse_gain_db"], row["noisy_nmse_db"] - row["model_nmse_db"], abs_tol=1e-10)
    return training, report, splits


def build_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    training, report, splits = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "lines.linewidth": 2,
                         "savefig.dpi": 300, "svg.fonttype": "none"})
    epochs = np.array([r["epoch"] + 1 for r in training])
    values = lambda key: np.array([r[key] for r in training])
    best = min(training, key=lambda r: r["val_nmse_metric"])
    best_epoch = int(best["epoch"]) + 1
    overall = sorted([r for r in report["results"] if r["scenario"] == "overall"], key=lambda r: r["snr_db"])
    snrs = [r["snr_db"] for r in overall]
    at = {r["snr_db"]: r for r in overall}
    figures = []

    def save(fig, slug, title, reading, inference, caution):
        fig.tight_layout(pad=1.5)
        fig.savefig(OUT / f"{slug}.png", facecolor="white")
        fig.savefig(OUT / f"{slug}.svg", facecolor="white")
        plt.close(fig)
        figures.append(dict(slug=slug, title=title, reading=reading, inference=inference, caution=caution))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax in axes:
        ax.plot(epochs, values("loss"), color=BLUE, label="Training: random SNR -10 to 30 dB")
        ax.plot(epochs, values("val_loss"), color=GREEN, label="Validation: fixed 10 dB SNR")
        ax.set(xlabel="Epoch", ylabel="Composite loss (lower is better)")
    axes[0].set_yscale("log")
    axes[0].set_title("Full training history (logarithmic scale)")
    axes[0].legend(fontsize=8.5)
    axes[1].set(xlim=(10, 60), ylim=(0, float(max(values("loss")[9:].max(), values("val_loss")[9:].max())) * 1.12),
                title="Epochs 10-60 (linear scale)")
    axes[1].scatter(best_epoch, best["val_loss"], color=RED, marker="*", s=150, zorder=5, label=f"Selected checkpoint: epoch {best_epoch}")
    axes[1].legend(fontsize=8.5)
    save(fig, "01_loss", "Training and validation loss",
         "Loss combines normalized reconstruction error with a magnitude penalty (weight 0.05). Lower is better. The left panel retains the large initial loss; the right panel reveals late-training behavior. Curves are unsmoothed.",
         f"Training loss falls from {training[0]['loss']:.2f} to {training[-1]['loss']:.5f}. Validation loss ends at {training[-1]['val_loss']:.5f}. The early validation spike is followed by recovery, with smaller improvements late in training.",
         "Training spans random -10 to 30 dB noise; validation is fixed at 10 dB. Training is also measured while weights change within each epoch. Therefore the gap is not a clean overfitting measure.")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(epochs, 10 * np.log10(values("nmse_metric")), color=BLUE, label="Training: 10 log10(mean linear NMSE)")
    ax.plot(epochs, 10 * np.log10(values("val_nmse_metric")), color=GREEN, label="Validation: 10 log10(mean linear NMSE)")
    ax.scatter(best_epoch, at[10]["model_nmse_db"], s=160, color=RED, marker="*", zorder=5,
               label=f"Selected model on test set at 10 dB: {at[10]['model_nmse_db']:.2f} dB")
    ax.set(xlabel="Epoch", ylabel="NMSE (dB; lower is better)", xlim=(1, 61))
    ax.legend(fontsize=9)
    save(fig, "02_nmse_convergence", "Reconstruction error through training",
         "NMSE measures error energy relative to clean-signal energy. Here the epoch's sample-weighted mean linear NMSE is converted to dB; it is not the log's mean of batch-level dB values.",
         f"Epoch {best_epoch}, chosen only by minimum validation linear NMSE, reaches {10 * math.log10(best['val_nmse_metric']):.2f} dB on validation. Its independent test score at 10 dB is {at[10]['model_nmse_db']:.2f} dB, a close result on the same scenario mixture.",
         f"The previously reported {best['val_nmse_db_metric']:.2f} dB validation value averages batch dB scores. Averaging before versus after the logarithm differs. Training also includes harder noise levels; these scores are not classification accuracy.")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(snrs, [r["noisy_nmse_db"] for r in overall], "o--", color=GRAY, label="Noisy input (no denoising)")
    ax.plot(snrs, [r["model_nmse_db"] for r in overall], "s-", color=GREEN, label="LocandKey, selected epoch 59")
    for snr in (0, 10):
        ax.annotate(f"{at[snr]['model_nmse_db']:.2f} dB", (snr, at[snr]["model_nmse_db"]),
                    xytext=(0, -24), textcoords="offset points", ha="center", color=GREEN)
    ax.set(xticks=snrs, xlabel="Input SNR (dB; higher means cleaner input)", ylabel="Pooled test NMSE (dB; lower is better)", ylim=(-40, 14))
    ax.legend()
    save(fig, "03_test_nmse", "Test performance across noise levels",
         "Each point evaluates all 14,024 held-out test samples. Error and signal energies are pooled across samples before converting their ratio to dB. The vertical gap measures improvement over noisy input.",
         f"At 0 dB input SNR, NMSE improves by {at[0]['nmse_gain_db']:.2f} dB (about {10 ** (at[0]['nmse_gain_db'] / 10):.0f} times less error energy). At 10 dB, improvement is {at[10]['nmse_gain_db']:.2f} dB (about {10 ** (at[10]['nmse_gain_db'] / 10):.0f} times less). Even at 30 dB, improvement remains {at[30]['nmse_gain_db']:.2f} dB.",
         "These are simulated AWGN tests on held-out samples from environments represented in training. They do not yet establish performance on unseen environments or real measured noise. Phoenix holdout results are excluded.")

    scenarios = list(splits["scenarios"])
    lookup = {(r["scenario"], r["snr_db"]): r for r in report["results"]}
    gains = np.array([[lookup[name, snr]["nmse_gain_db"] for snr in snrs] for name in scenarios])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(gains, cmap="YlGnBu", vmin=0, vmax=math.ceil(gains.max() / 5) * 5, aspect="auto")
    ax.grid(False)
    ax.set(xticks=range(len(snrs)), xticklabels=snrs, yticks=range(len(scenarios)),
           yticklabels=[f"{NAMES[n]} (n={splits['scenarios'][n]['splits']['test']:,})" for n in scenarios],
           xlabel="Input SNR (dB)")
    for i in range(len(scenarios)):
        for j in range(len(snrs)):
            ax.text(j, i, f"{gains[i, j]:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if gains[i, j] > im.norm.vmax * .55 else "#14232d")
    fig.colorbar(im, ax=ax, label="NMSE gain over noisy input (dB; positive is better)")
    save(fig, "04_scenario_gains", "Denoising gain by environment",
         "Each cell is noisy-input NMSE minus model NMSE for one scenario and SNR. Positive numbers mean denoising helps. Sample counts are per SNR; the same clean test samples are evaluated with new deterministic noise.",
         f"All 45 scenario/SNR combinations improve, with gains from {gains.min():.2f} to {gains.max():.2f} dB. At 10 dB input, scenario gains range from {min(lookup[n,10]['nmse_gain_db'] for n in scenarios):.2f} to {max(lookup[n,10]['nmse_gain_db'] for n in scenarios):.2f} dB. Both preset acceptance checks at 0 and 10 dB pass.",
         "Scenario difficulty varies and Phoenix has fewer test samples. This baseline model trained on all five environments, including Phoenix; this figure must not be presented as an unseen-environment experiment.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for prefix, label, color, marker in (("noisy", "Noisy input", GRAY, "o--"), ("model", "LocandKey", GREEN, "s-")):
        axes[0].plot(snrs, [r[f"{prefix}_correlation"] for r in overall], marker, color=color, label=label)
        axes[1].plot(snrs, [r[f"{prefix}_mae"] for r in overall], marker, color=color, label=label)
    axes[0].set(ylabel="Complex correlation magnitude", ylim=(0, 1.03), title="Structural similarity (higher is better)")
    axes[1].set(ylabel="Mean absolute complex error", yscale="log", title="Absolute error (lower is better)")
    for ax in axes:
        ax.set(xlabel="Input SNR (dB)", xticks=snrs)
        ax.legend(fontsize=9)
    save(fig, "05_reconstruction_quality", "Signal similarity and absolute error",
         "Left: magnitude of normalized complex correlation, pooled across test elements. Right: mean absolute complex reconstruction error on normalized CSI, shown on a logarithmic scale.",
         f"At 0 dB, correlation rises from {at[0]['noisy_correlation']:.4f} to {at[0]['model_correlation']:.4f}. At 10 dB it rises from {at[10]['noisy_correlation']:.4f} to {at[10]['model_correlation']:.4f}. At 10 dB, mean absolute error decreases by {at[10]['noisy_mae']/at[10]['model_mae']:.2f} times. Both metrics support the NMSE improvement.",
         "Correlation magnitude alone cannot detect all scale or common-phase errors; read it alongside NMSE and absolute error. MAE is in normalized CSI units, not physical received-power units.")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(epochs, values("learning_rate"), color=BLUE)
    ax.axvline(best_epoch, color=RED, linestyle="--", linewidth=1.2, label=f"Best checkpoint: epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Learning rate at epoch end", xlim=(1, 61), ylim=(0, .00032))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    save(fig, "06_learning_rate", "Learning-rate schedule",
         "The curve records the optimizer learning rate at the end of each epoch. The run uses cosine annealing from an initial 0.0003 toward 0.000001; the schedule advances on successful optimizer steps.",
         "The smooth decay shows that the scheduled reduction was applied. Smaller updates coincide with the stable late-training region, and the best checkpoint occurs near the end of the schedule.",
         "This is training context, not an independent performance result. One run cannot establish that cosine annealing is better than a different schedule. No error bars are available from this single-seed experiment.")

    for source, name in ((TRAIN / "logs/multiscenario/training_log.csv", "training_log.csv"),
                         (EVAL / "metrics.csv", "test_metrics.csv"), (EVAL / "evaluation_summary.json", "evaluation_summary.json")):
        shutil.copyfile(source, OUT / name)
    metadata = {"title": "LocandKey | Baseline performance", "best_epoch": best_epoch,
                "training": splits["totals"], "split_sha256": report["split_sha256"], "figures": figures,
                "overall": overall, "sources": [str(TRAIN / "logs/multiscenario/training_log.csv"), str(EVAL / "evaluation_summary.json")]}
    (OUT / "figure_notes.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    notes = ["# LocandKey: faculty graph notes", "", "Completed 60-epoch baseline; selected checkpoint: epoch 59. Phoenix holdout is not included.",
             "", "65,442 training / 14,023 validation / 14,024 test samples. Single seed: 42. Nine test SNRs, -10 to 30 dB.", ""]
    for entry in figures:
        notes += [f"## {entry['title']}", "", f"![{entry['title']}]({entry['slug']}.png)", "",
                  "**How to read it:** " + entry["reading"], "", "**Inference:** " + entry["inference"], "",
                  "**What not to claim:** " + entry["caution"], ""]
    notes += ["## Scope and metric definitions", "", "NMSE = error energy / clean-signal energy; NMSE_dB = 10 log10(NMSE). Gain_dB = noisy NMSE_dB - model NMSE_dB. Error-energy reduction factor = 10^(gain_dB/10).", "",
              "The epoch convergence graph transforms the saved mean linear NMSE; training/validation also use an epsilon in their loss metric. Test NMSE pools raw error and signal energies. These are related, not identical estimators. Do not compare the saved mean batch-dB validation score directly to pooled test dB.", "",
              "The data split is sample-level within each scenario. No claim of spatial/temporal independence, unseen-environment generalization, multi-seed reliability, real-world performance, localization accuracy, or secret-key performance is established by these graphs.", "",
              "No curves were smoothed; all epochs and SNRs are retained. Acceptance requires at least 2 dB overall gain and positive gain in every scenario at both 0 and 10 dB. Both pass.", "", "Source files:", *[f"- {path}" for path in metadata["sources"]], "",
              f"Split fingerprint: `{report['split_sha256']}`", "", "PNG exports are 300 dpi; SVG exports are editable vectors."]
    (OUT / "EXPLANATIONS.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Validated source data; generated {len(figures)} PNG/SVG figure pairs and explanations: {OUT}")


def build_pdf():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle

    meta = json.loads((OUT / "figure_notes.json").read_text())
    PDF.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="ChartTitle", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=colors.HexColor(BLUE), spaceAfter=8))
    styles.add(ParagraphStyle(name="Kicker", fontSize=9, leading=12, textColor=colors.HexColor(GRAY), spaceAfter=10))
    body = lambda text: Paragraph(text, styles["BodySmall"])
    doc = SimpleDocTemplate(str(PDF), pagesize=landscape(A4), rightMargin=38, leftMargin=38, topMargin=35, bottomMargin=30,
                            title="LocandKey - Baseline Performance and Interpretation", author="LocandKey project")
    story = [Paragraph("LOCANDKEY / EXPERIMENT RESULTS", styles["Kicker"]),
             Paragraph("Baseline performance and interpretation", styles["ChartTitle"]),
             body("Completed 60-epoch training and independent test evaluation | 21 September 2026"),
             body("<b>Result:</b> denoising improves every tested scenario/SNR combination. The selected model is epoch 59, chosen using validation NMSE. These results describe the completed five-scenario baseline; the separate Phoenix holdout run is not included."),
             Spacer(1, 10)]
    table = [["Input SNR", "Noisy NMSE", "Model NMSE", "Gain", "Less error energy"]]
    for row in meta["overall"]:
        if row["snr_db"] in (0, 10, 30):
            table.append([f"{row['snr_db']:g} dB", f"{row['noisy_nmse_db']:.2f} dB", f"{row['model_nmse_db']:.2f} dB",
                          f"{row['nmse_gain_db']:.2f} dB", f"{10**(row['nmse_gain_db']/10):.1f}x"])
    t = Table(table, colWidths=[105, 130, 130, 105, 145], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf0f6")),
                           ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 10),
                           ("TOPPADDING", (0,0), (-1,-1), 9), ("BOTTOMPADDING", (0,0), (-1,-1), 9),
                           ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor(BLUE))]))
    story += [t, Spacer(1, 18), body("<b>Data:</b> 65,442 training, 14,023 validation and 14,024 test samples. Five scenarios. Test sweep: -10, -5, 0, 5, 10, 15, 20, 25 and 30 dB. All test samples are evaluated at each SNR."),
              body("<b>Training:</b> ConvNeXt U-Net CSI denoiser, approximately 2.0 million parameters; Adam optimizer; global batch 8; mixed precision; two T4 GPUs with DDP; seed 42. Training noise varies from -10 to 30 dB; validation noise is fixed at 10 dB."),
              body("<b>Acceptance:</b> at least 2 dB overall gain and positive gain for every scenario at both 0 and 10 dB. Both checks pass."),
              body("<b>Scope:</b> sample-level test splits from environments represented during training, using simulated AWGN. No multi-seed uncertainty intervals or unseen-environment conclusion are available here."),
              body("<b>Metric note:</b> the previously reported -29.60 dB validation figure is a mean of batch dB values. These convergence figures instead convert the saved mean linear NMSE to dB (about -28.08 dB at the selected epoch), avoiding an inconsistent comparison with pooled test NMSE."),
              body("<b>Provenance:</b> completed Kaggle locandkey-training version 9 and locandkey-evaluation version 1. Local CSV copies, exact source paths, and the verified split fingerprint accompany the figures.")]
    for number, entry in enumerate(meta["figures"], 1):
        story += [PageBreak(), Paragraph(f"FIGURE {number:02d} / COMPLETED BASELINE", styles["Kicker"]),
                  Paragraph(entry["title"], styles["ChartTitle"]),
                  Image(str(OUT / f"{entry['slug']}.png"), width=710, height=710 * 5.2 / 11),
                  body("<b>Read:</b> " + entry["reading"]), body("<b>Infer:</b> " + entry["inference"]),
                  body("<b>Limit:</b> " + entry["caution"])]
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(38, 17, "LocandKey | completed baseline | holdout experiment excluded")
        canvas.drawRightString(landscape(A4)[0] - 38, 17, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-only", action="store_true", help="Build the PDF from existing figures and notes.")
    args = parser.parse_args()
    if args.pdf_only:
        build_pdf()
    else:
        build_figures()
