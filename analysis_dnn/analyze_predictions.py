"""
Load regression_predictions_*_tones.csv in this folder.
X: note index from filename; Y: true_elev - pred_elev.

Run as a script, or paste into a REPL from the top. If __file__ is missing (REPL),
set cwd to this directory (analysis_dnn) so glob finds the CSVs.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

try:
    DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DIR = os.getcwd()
    _sub = os.path.join(DIR, "analysis_dnn")
    if not glob.glob(
        os.path.join(DIR, "regression_predictions_*_tones.csv")
    ) and os.path.isdir(_sub):
        DIR = _sub

ROOT = os.path.dirname(DIR)
PLOT_DIR = os.path.join(ROOT, "plots")

PALETTE = {
    "flute": "#e22c1f",
    "harmoniccomplex": "#E5A09C",
    "viola": "#2e33a6",
    "viola_complex": "#6a5acd",
}


def note_from_fname(name):
    s = str(name).strip()
    m = re.search(r"stim_(\d+)_", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\.wav$", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise ValueError(f"No note index in filename: {name!r}")


paths = sorted(glob.glob(os.path.join(DIR, "regression_predictions_*_tones.csv")))
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
    df["elev_error"] = df["true_elev"] - df["pred_elev"]
    df["source_file"] = base
    frames.append(df)

all_pred = pandas.concat(frames, ignore_index=True)
print(all_pred.groupby("condition").size())

corr_rows = []
for cond, g in all_pred.groupby("condition", sort=False):
    x = g["midi_note"]
    y = g["elev_error"]
    valid = x.notna() & y.notna()
    xv, yv = x[valid], y[valid]
    n = int(valid.sum())
    r = xv.corr(yv, method="pearson") if n >= 2 else float("nan")
    p = float("nan")
    if n >= 3 and not pandas.isna(r):
        try:
            from scipy.stats import pearsonr

            _, p = pearsonr(xv.to_numpy(), yv.to_numpy())
        except ImportError:
            pass
    corr_rows.append(
        {"condition": cond, "n": n, "pearson_r": r, "p_two_sided": p}
    )
corr_df = pandas.DataFrame(corr_rows)
print("\nPearson correlation (MIDI note vs. elevation error) by condition:")
with pandas.option_context("display.max_columns", None, "display.width", 120):
    print(corr_df.to_string(index=False))

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
axes[0].set_ylabel("True elevation − predicted elevation")
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
