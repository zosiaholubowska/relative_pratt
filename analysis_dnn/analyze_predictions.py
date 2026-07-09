"""
Load regression_predictions_*_tones.csv in this folder.
X: note index from filename; Y: pred_elev - true_elev (positive = overestimate).

"""
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"
DIR = os.getcwd()
PLOT_DIR = os.path.join(DIR, "plots")

PALETTE = {
    "flute": "#e22c1f",
    "harmoniccomplex": "#E5A09C",
    "viola": "#2e33a6",
    "viola_complex": "#9FA0CC",
    "piano": "#E5A09C",
}

def midi_to_hz(midi):
    """Equal temperament: A4 (MIDI 69) = 440 Hz."""
    return 440.0 * (2.0 ** ((pandas.Series(midi, dtype=float) - 69.0) / 12.0))


def note_from_fname(name):
    s = str(name).strip()
    m = re.search(r"stim_(\d+)_", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\.wav$", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise ValueError(f"No note index in filename: {name!r}")


paths = sorted(glob.glob(os.path.join(DIR, "analysis_dnn", "regression_predictions_*_tones.csv")))
if not paths:
    raise FileNotFoundError(f"No regression_predictions_*_tones.csv in {DIR}")

frames = []
for path in paths:
    base = os.path.basename(path)
    m = re.match(r"regression_predictions_(.+)_tones\.csv$", base, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unexpected filename: {base!r}")
    cond = m.group(1).lower()
    df = pandas.read_csv(path)
    need = {"filename", "true_elev", "pred_elev"}
    if need - set(df.columns):
        raise ValueError(f"{path}: missing columns {need - set(df.columns)}")
    df["condition"] = cond
    df["midi_note"] = df["filename"].map(note_from_fname)
    df["frequency_hz"] = midi_to_hz(df["midi_note"])
    df["elev_error"] = df["pred_elev"] - df["true_elev"]
    df["source_file"] = base
    frames.append(df)

all_pred = pandas.concat(frames, ignore_index=True)
print(all_pred.groupby("condition").size())

corr_rows_midi = []
corr_rows_freq = []
for cond, g in all_pred.groupby("condition", sort=False):
    y = g["elev_error"]
    valid_m = g["midi_note"].notna() & y.notna()
    xm, ym = g.loc[valid_m, "midi_note"], y[valid_m]
    valid_f = g["frequency_hz"].notna() & y.notna()
    xf, yf = g.loc[valid_f, "frequency_hz"], y[valid_f]
    n = int(valid_m.sum())
    r_m = xm.corr(ym, method="pearson") if n >= 2 else float("nan")
    r_f = xf.corr(yf, method="pearson") if len(xf) >= 2 else float("nan")
    p_m = float("nan")
    p_f = float("nan")
    if n >= 3 and not pandas.isna(r_m):
        try:
            from scipy.stats import pearsonr

            _, p_m = pearsonr(xm.to_numpy(), ym.to_numpy())
        except ImportError:
            pass
    if len(xf) >= 3 and not pandas.isna(r_f):
        try:
            from scipy.stats import pearsonr

            _, p_f = pearsonr(xf.to_numpy(), yf.to_numpy())
        except ImportError:
            pass
    corr_rows_midi.append(
        {"condition": cond, "n": n, "pearson_r": r_m, "p_two_sided": p_m}
    )
    corr_rows_freq.append(
        {"condition": cond, "n": n, "pearson_r": r_f, "p_two_sided": p_f}
    )
corr_df_midi = pandas.DataFrame(corr_rows_midi)
corr_df_freq = pandas.DataFrame(corr_rows_freq)
print("\nPearson correlation (MIDI note vs. elevation error) by condition:")
with pandas.option_context("display.max_columns", None, "display.width", 120):
    print(corr_df_midi.to_string(index=False))
print("\nPearson correlation (f₀ vs. elevation error) by condition:")
with pandas.option_context("display.max_columns", None, "display.width", 120):
    print(corr_df_freq.to_string(index=False))

os.makedirs(PLOT_DIR, exist_ok=True)
order = sorted(all_pred["midi_note"].unique())
x_pad = 1.0

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

sns.scatterplot(
    data=all_pred,
    x="midi_note",
    y="elev_error",
    hue="condition",
    palette=PALETTE,
    alpha=0.5,
    s=14,
    ax=axes[0],
)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
axes[0].set_ylabel("Predicted elevation − true elevation")
axes[0].set_title("Per-trial elevation error vs. MIDI note (DNN predictions)")
axes[0].legend(title="Condition")

sns.pointplot(
    data=all_pred,
    x="midi_note",
    y="elev_error",
    hue="condition",
    palette=PALETTE,
    dodge=0.25,
    errorbar=("ci", 95),
    ax=axes[1],
    order=order,
)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
axes[1].set_xlabel("MIDI note (from filename)")
axes[1].set_ylabel("Mean error (95% CI)")
axes[1].legend(title="Condition")
axes[0].set_xlim(min(order) - x_pad, max(order) + x_pad)

plt.tight_layout()
out_png = os.path.join(PLOT_DIR, "dnn_elevation_error_by_note.png")
out_svg = os.path.join(PLOT_DIR, "dnn_elevation_error_by_note.svg")
plt.savefig(out_png, dpi=200)
plt.savefig(out_svg)
plt.close()
print(f"Wrote {out_png}")
print(f"Wrote {out_svg}")

# Single-panel summary: mean error vs. f₀ (same estimator style as lower panel of main figure)
all_pred["f0_hz"] = all_pred["frequency_hz"].round(0).astype(int)
freq_order = sorted(all_pred["f0_hz"].unique())
fig2, ax_f = plt.subplots(1, 1, figsize=(10, 6))
sns.pointplot(
    data=all_pred,
    x="f0_hz",
    y="elev_error",
    hue="condition",
    palette=PALETTE,
    dodge=0.25,
    errorbar=("ci", 95),
    ax=ax_f,
    order=freq_order,
)
n_xt = len(freq_order)
if n_xt > 4:
    tick_idx = [int(round(k * (n_xt - 1) / 3)) for k in range(4)]
    tick_idx = list(dict.fromkeys(tick_idx))
    ax_f.set_xticks(tick_idx)
    ax_f.set_xticklabels([str(freq_order[i]) for i in tick_idx])
ax_f.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax_f.set_xlabel("Fundamental frequency (Hz)")
ax_f.set_ylabel("Mean error (95% CI)")
ax_f.set_title("DNN elevation error vs. f₀")
leg = ax_f.get_legend()
if leg is not None: 
    leg.set_title("Condition")
    leg.set_frame_on(False)
    leg.set_bbox_to_anchor((1.02, 1.0))
    leg.set_loc("upper left")
plt.tight_layout()
out_freq_png = os.path.join(PLOT_DIR, "dnn_elevation_error_by_frequency.png")
out_freq_svg = os.path.join(PLOT_DIR, "dnn_elevation_error_by_frequency.svg")
plt.savefig(out_freq_png, dpi=200, bbox_inches="tight")
plt.savefig(out_freq_svg, bbox_inches="tight")
plt.close()
print(f"Wrote {out_freq_png}")
print(f"Wrote {out_freq_svg}")

# OLS regression lines only (no scatter), cf. analysis.py elevation_diff_filtered
fig3, ax_reg = plt.subplots(figsize=(6, 6))
for condition in PALETTE:
    subset = all_pred[all_pred["condition"] == condition]
    if subset.empty:
        continue
    sns.regplot(
        x="frequency_hz",
        y="elev_error",
        data=subset,
        color=PALETTE[condition],
        scatter=False,
        ax=ax_reg,
        label=condition,
    )
ax_reg.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax_reg.set_xlabel("Frequency (Hz)", fontsize=12)
ax_reg.set_ylabel("Predicted elevation − true elevation", fontsize=12)
ax_reg.set_title("DNN elevation error vs. frequency (linear fit per condition)")
ax_reg.legend(title="Condition")
plt.tight_layout()
out_reg_png = os.path.join(PLOT_DIR, "dnn_elevation_error_freq_regression.png")
out_reg_svg = os.path.join(PLOT_DIR, "dnn_elevation_error_freq_regression.svg")
plt.savefig(out_reg_png, dpi=200, bbox_inches="tight")
plt.savefig(out_reg_svg, bbox_inches="tight")
plt.close()
print(f"Wrote {out_reg_png}")
print(f"Wrote {out_reg_svg}")
