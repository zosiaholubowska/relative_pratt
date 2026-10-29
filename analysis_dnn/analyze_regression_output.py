import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from scipy.stats import pearsonr, spearmanr

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams["figure.dpi"] = 300

DIR = os.getcwd()
REGRESSION_DIR = os.path.join(DIR, "analysis_dnn", "regression_output")
PLOT_DIR = os.path.join(DIR, "analysis_dnn", "plots")


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
    os.path.join(DIR, "acoustic_analysis", "spectral_centroid_stimuli.csv")
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

# Correlation between centroid and elevation difference
print("\nCorrelation between centroid and elevation difference:")
for model in per_sound["model"].unique():
    df_model = per_sound[per_sound["model"] == model]
    r,p = pearsonr(df_model["centroid_khz"], df_model["elevation_diff"])
    print(f"  {model}: Pearson's r = {r:.3f}, p = {p:.3g}")
    r,p = spearmanr(df_model["centroid_khz"], df_model["elevation_diff"])
    print(f"  {model}: Spearman's r = {r:.3f}, p = {p:.3g}")

# ------------------------------------------------------------
# Plot the frequency - elevation mapping
# ------------------------------------------------------------

sns.set_context("notebook")

plot_name = input("Enter the name of the plot:")

MODEL_LABELS = {"prob_bias": "World Bias", "flipped": "Flipped HRTF", "basic": "Basic"}
MODEL_COLORS = {"Basic": "#1a80bb", "Flipped HRTF": "#8cc5e3", "World Bias": "#a00000"}
per_sound["model_label"] = per_sound["model"].map(
    lambda m: next((label for key, label in MODEL_LABELS.items() if key in m), m)
)

g = sns.lmplot(
    data=per_sound,
    x="centroid_khz",
    y="elevation_diff",
    hue="model_label",
    scatter_kws={"s": 18, "alpha": 0.45},
    palette=MODEL_COLORS,
    aspect=1,
    legend=False,
    ci=95
)
g.set_axis_labels("Spectral centroid (kHz)", "Elevation error (pred − true, °)")
plt.tight_layout()
plt.legend(loc='lower right', fontsize='xx-small')
plt.savefig(os.path.join(PLOT_DIR, f"centroid_elevation_error_{plot_name}.png"))
plt.show()

# ------------------------------------------------------------------------------------------------
# Compute the correlation with f0 MIDI
# ------------------------------------------------------------------------------------------------

midi_analysis = input("Do you want to analyze the correlation with f0 MIDI? (y/n):")
if midi_analysis == "y":
    df["midi_note"] = df["filename"].str.extract(r"stim_(\d+)_", expand=False).astype(float)
    per_sound["midi_note"] = per_sound["filename"].str.extract(r"stim_(\d+)_", expand=False).astype(float)

    print("\nCorrelation between f0 MIDI and elevation difference:")
    for model in per_sound["model"].unique():
        df_model = per_sound[per_sound["model"] == model]
        r,p = pearsonr(df_model["midi_note"], df_model["elevation_diff"])
        print(f"  {model}: Pearson's r = {r:.3f}, p = {p:.3g}")

