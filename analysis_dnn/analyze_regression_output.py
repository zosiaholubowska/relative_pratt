import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["figure.dpi"] = 300

DIR = os.getcwd()
REGRESSION_DIR = os.path.join(DIR, "analysis_dnn", "regression_output")
PLOT_DIR = os.path.join(os.path.dirname(DIR), "plots")


file_names_input = input("Enter specific regression output file names to analyze (comma-separated):")
file_suffixes = [s.strip() for s in file_names_input.split(",") if s.strip()]

selected_files = [os.path.join(REGRESSION_DIR, f"regression_predictions_{s}.csv") for s in file_suffixes]

existing_files = []
missing_files = []
for f in selected_files:
    if os.path.isfile(f):
        existing_files.append(f)
    else:
        missing_files.append(f)

if missing_files:
    print("\nWARNING: The following files were NOT found and will be skipped:")
    for mf in missing_files:
        print(" -", os.path.basename(mf))
if not existing_files:
    print("No valid regression output files were provided. Exiting.")

print("\nFiles to be analyzed:")
for ef in existing_files:
    print(" -", os.path.basename(ef))

dfs = [pd.read_csv(f).assign(model=os.path.basename(f).removeprefix("regression_predictions_").removesuffix(".csv"))
    for f in existing_files]
df = pd.concat(dfs, ignore_index=True)

centroids = pd.read_csv(
    os.path.join(DIR, "acoustic_analysis", "spectral_centroid_training_stimuli.csv")
).set_index("sound")[["centroid", "centroid_midi"]]
df["centroid"] = df["filename"].map(centroids["centroid"])
df["centroid_midi"] = df["filename"].map(centroids["centroid_midi"])
df["elevation_diff"] = df["pred_elev"] - df["true_elev"]

# ------------------------------------------------------------
# Centroid → elevation error, and whether the slope differs by model.
# One row per (model, sound): mean error over all azim/elev trials for that sound.
# Filename only links centroid to the sound; it is not a random effect here.
# ------------------------------------------------------------

per_sound = (
    df.groupby(["model", "filename"], as_index=False)
    .agg(
        centroid=("centroid", "first"),
        centroid_midi=("centroid_midi", "first"),
        elevation_diff=("elevation_diff", "mean"),
    )
    .dropna(subset=["centroid", "elevation_diff"])
)
per_sound["centroid_khz"] = per_sound["centroid"] / 1000.0

print("\nPer model (sound means, n = one point per stimulus):")
for model, g in per_sound.groupby("model", sort=False):
    fit = smf.ols("elevation_diff ~ centroid_khz", data=g).fit()
    r = g["centroid_khz"].corr(g["elevation_diff"])
    print(
        f"  {model}: slope = {fit.params['centroid_khz']:.3f} °/kHz, "
        f"intercept = {fit.params['Intercept']:.3f} °, r = {r:.3f}, p(slope) = {fit.pvalues['centroid_khz']:.3g}"
    )

print("\nCombined: elevation_diff ~ centroid_khz * C(model)")
ols = smf.ols("elevation_diff ~ centroid_khz * C(model)", data=per_sound).fit(
    cov_type="cluster",
    cov_kwds={"groups": per_sound["filename"]},
)
print(ols.summary())

ref = sorted(per_sound["model"].unique())[0]
print(f"\nReference model for interactions: {ref!r}")
print(f"  slope {ref}: {ols.params['centroid_khz']:.3f} °/kHz")
for name, param in ols.params.items():
    if name.startswith("centroid_khz:C(model)"):
        level = name.split("[T.")[1].rstrip("]")
        total = ols.params["centroid_khz"] + param
        print(f"  slope {level}: {total:.3f} °/kHz (Δ vs {ref}: {param:+.3f}, p = {ols.pvalues[name]:.3g})")

# ------------------------------------------------------------
# Plot the frequency - elevation mapping
# ------------------------------------------------------------

sns.set_context("notebook")

g = sns.lmplot(
    data=per_sound,
    x="centroid_khz",
    y="elevation_diff",
    hue="model",
    scatter_kws={"s": 18, "alpha": 0.45},
    palette='tab10',
    height=6,
    aspect=1.6,
    legend=True,
    ci=95
)
g.set_axis_labels("Spectral centroid (kHz)", "Elevation error (pred − true, °)")
plt.tight_layout()
plt.show()
