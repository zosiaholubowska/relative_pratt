"""
Load predictions_*.csv from this folder, one row per model prediction.
X: MIDI note parsed from the stimulus filename; Y: true_elev - pred_elev.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
PLOT_DIR = os.path.join(ROOT, "plots")

# Match analysis.py styling
PALETTE = {
    "flute": "#e22c1f",
    "harmonic": "#E5A09C",
    "viola": "#2e33a6",
}


def midi_note_from_filename(name: str) -> int:
    """Extract the trailing note number before .wav (e.g. chord_flute_55.wav -> 55)."""
    m = re.search(r"(\d+)\.wav$", str(name).strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"No MIDI note found in filename: {name!r}")
    return int(m.group(1))


def condition_from_predictions_path(path: str) -> str:
    """predictions_flute.csv -> flute; predictions_harmonic.csv -> harmonic."""
    base = os.path.basename(path)
    m = re.match(r"predictions_(.+)\.csv$", base, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unexpected predictions file name: {base!r}")
    return m.group(1).lower()


def load_all_predictions() -> pandas.DataFrame:
    paths = sorted(glob.glob(os.path.join(DIR, "predictions_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No predictions_*.csv under {DIR}")

    frames = []
    for path in paths:
        cond = condition_from_predictions_path(path)
        df = pandas.read_csv(path)
        required = {"filename", "true_elev", "pred_elev"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        df["condition"] = cond
        df["midi_note"] = df["filename"].map(midi_note_from_filename)
        # Residual: positive => model predicted elevation lower than truth
        df["elev_error"] = df["true_elev"] - df["pred_elev"]
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    return pandas.concat(frames, ignore_index=True)


def plot_elevation_error_by_note(data: pandas.DataFrame) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    order = sorted(data["midi_note"].unique())
    x_pad = 1.0

    # Top: numeric MIDI on x. Bottom: categorical pointplot (even spacing between notes).
    # Do not share x: categorical bottom + numeric top hid scatter off-screen with sharex=True.
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    sns.scatterplot(
        data=data,
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
        data=data,
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


if __name__ == "__main__":
    all_pred = load_all_predictions()
    print(all_pred.groupby("condition").size())
    plot_elevation_error_by_note(all_pred)
