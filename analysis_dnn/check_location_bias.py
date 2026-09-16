"""
Check whether the centroid -> elevation bias made it into the rendered datasets.

Joins the spectral-centroid cache written by probability_bias.py with the
(stimulus, azim, elev) tables dumped by inspect_cochleagram_tfrecord.py, and
plots every presentation plus the mean elevation per stimulus against centroid.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["svg.fonttype"] = "none"

if "__file__" in globals():
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:  
    DIR = os.getcwd()
RESULTS_DIR = f"{DIR}/Results"
PLOT_DIR = f"{DIR}/plots"
LOCATIONS_DIR = f"{DIR}/analysis_dnn/Results"

CENTROID_CACHE = f"{RESULTS_DIR}/naturalsounds165_spectral_centroids.csv"
LOCATIONS_TEMPLATE = f"{LOCATIONS_DIR}/stimulus_locations_{{split}}.csv"

SPLITS = ("test", "train")

# Elevation range the bias was designed over (see probability_bias.py).
ELEV_MIN = 0.0
ELEV_MAX = 60.0

os.makedirs(PLOT_DIR, exist_ok=True)


def centroid_to_target_elevation(centroid_hz, cent_min, cent_max, elev_min, elev_max):
    """Intended mapping from probability_bias.py, used as the reference line."""
    centroid_hz = np.clip(centroid_hz, cent_min, cent_max)
    t = (centroid_hz - cent_min) / (cent_max - cent_min)
    return elev_min + t * (elev_max - elev_min)


def plot_split(split, centroid_df, cent_min, cent_max):
    path = LOCATIONS_TEMPLATE.format(split=split)
    if not os.path.isfile(path):
        print(f"{split}: {path} not found, skipping")
        return

    trials = pd.read_csv(path).merge(
        centroid_df, on="stimulus", how="inner", validate="many_to_one"
    )
    per_stim = (
        trials.groupby("stimulus")
        .agg(
            spectral_centroid_hz=("spectral_centroid_hz", "first"),
            mean_elev=("elev", "mean"),
        )
        .reset_index()
    )
    fit = stats.linregress(per_stim["spectral_centroid_hz"], per_stim["mean_elev"])
    print(
        f"{split}: {len(trials)} presentations of {len(per_stim)} stimuli, "
        f"mean elevation = {fit.slope * 1000:.2f}°/kHz, r = {fit.rvalue:.3f}, p = {fit.pvalue:.3g}"
    )

    fig, ax = plt.subplots(figsize=(12 / 2.54, 10 / 2.54))
    ax.scatter(
        trials["spectral_centroid_hz"],
        trials["elev"] + np.random.default_rng(0).uniform(-3, 3, len(trials)),
        s=4,
        color="0.75",
        alpha=0.15,
        linewidths=0,
        zorder=1,
        label="Single presentations (jittered)",
    )
    ax.scatter(
        per_stim["spectral_centroid_hz"],
        per_stim["mean_elev"],
        s=26,
        c=per_stim["mean_elev"],
        cmap="inferno",
        vmin=ELEV_MIN,
        vmax=ELEV_MAX,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
        label="Mean elevation per stimulus",
    )
    c_line = np.linspace(cent_min, cent_max, 100)
    ax.plot(
        c_line,
        centroid_to_target_elevation(c_line, cent_min, cent_max, ELEV_MIN, ELEV_MAX),
        "k--",
        linewidth=1.2,
        zorder=4,
        label="Intended target elevation",
    )
    ax.plot(
        c_line,
        fit.intercept + fit.slope * c_line,
        color="#e22c1f",
        linewidth=1.8,
        zorder=5,
        label="Fit to stimulus means",
    )
    ax.set_xlabel("Spectral centroid (Hz)")
    ax.set_ylabel("Presentation elevation (°)")
    ax.set_title(
        f"Elevation vs centroid ({split} set)\n"
        f"r = {fit.rvalue:.3f}, {fit.slope * 1000:.2f}°/kHz",
        fontsize=10,
    )
    ax.set_ylim(ELEV_MIN - 5, ELEV_MAX + 5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/location_bias_centroid_elevation_{split}.svg", dpi=300)
    fig.savefig(f"{PLOT_DIR}/location_bias_centroid_elevation_{split}.png", dpi=300)
    plt.close(fig)


centroid_df = pd.read_csv(CENTROID_CACHE)
cent_min = float(centroid_df["spectral_centroid_hz"].min())
cent_max = float(centroid_df["spectral_centroid_hz"].max())

for split in SPLITS:
    plot_split(split, centroid_df, cent_min, cent_max)
