"""
Analyse regression predictions for Pratt test sounds.

Sound types: pure, bright harmonic, dark harmonic, ERB pink noise.
Plots mean elevation error vs. frequency (log scale) with SEM and Pearson r.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

plt.rcParams["svg.fonttype"] = "none"

DIR = os.path.dirname(os.path.abspath(__file__))
REGRESSION_DIR = os.path.join(DIR, "regression_output")
PLOT_DIR = os.path.join(os.path.dirname(DIR), "plots")

SOUND_TYPES = ("pure", "bright", "dark", "noise")
SOUND_TYPE_LABELS = {
    "pure": "Pure tone",
    "bright": "Bright harmonic",
    "dark": "Dark harmonic",
    "noise": "ERB pink noise",
}
SOUND_TYPE_COLORS = dict(zip(SOUND_TYPES, sns.color_palette("Set1", len(SOUND_TYPES))))

MODEL_TYPE_LABELS = {
    "hrtf_flipped": "HRTF flipped",
    "test": "Test",
    "tones": "Test",
}

REQUIRED_COLS = {"filename", "true_azim", "true_elev", "pred_azim", "pred_elev"}


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((pd.Series(midi, dtype=float) - 69.0) / 12.0))


def build_erb_band_center_hz(samplerate=44100):
    centers = {}
    try:
        from slab.filter import Filter

        filter_params = {"n_filters": 10, "low_cutoff": 100}
        center_erb, _, erb_spacing = Filter._center_freqs(
            low_cutoff=filter_params["low_cutoff"],
            high_cutoff=samplerate / 2,
            bandwidth=filter_params.get("bandwidth", 1 / 3),
            pass_bands=False,
            n_filters=filter_params["n_filters"],
        )
        band_low_hz = Filter._erb2freq(center_erb - erb_spacing)
        band_high_hz = Filter._erb2freq(center_erb + erb_spacing)
        for band in range(len(band_low_hz)):
            center = float(np.sqrt(band_low_hz[band] * band_high_hz[band]))
            centers[f"erb_pink_band_{band}.wav"] = center
    except ImportError:
        for band, center in enumerate(np.geomspace(150.0, 12000.0, 10)):
            centers[f"erb_pink_band_{band}.wav"] = float(center)
    return centers


ERB_BAND_CENTER_HZ = build_erb_band_center_hz()


def sound_type_from_filename(name):
    s = str(name).lower()
    if s.startswith("erb_pink_band"):
        return "noise"
    if "_pure.wav" in s:
        return "pure"
    if "_bright_harmonic.wav" in s:
        return "bright"
    if "_dark_harmonic.wav" in s:
        return "dark"
    raise ValueError(f"Unknown Pratt sound filename: {name!r}")


def midi_from_filename(name):
    m = re.search(r"stim_(\d+)_", str(name), re.IGNORECASE)
    return int(m.group(1)) if m else np.nan


def plot_freq_hz_from_row(row):
    if row["sound_type"] == "noise":
        return ERB_BAND_CENTER_HZ.get(str(row["filename"]), np.nan)
    return row["frequency_hz"]


def load_pratt_predictions(regression_dir=REGRESSION_DIR):
    paths = sorted(glob.glob(os.path.join(regression_dir, "regression_prediction*_pratt_sounds_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No regression_prediction*_pratt_sounds_*.csv in {regression_dir}")

    frames = []
    for path in paths:
        base = os.path.basename(path)
        m = re.match(
            r"regression_predictions?_pratt_sounds_(?P<model_type>.+)\.csv$",
            base,
            re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"Unexpected filename: {base!r}")
        model_type = m.group("model_type").lower()

        df = pd.read_csv(path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        df["model_type"] = model_type
        df["source_file"] = base
        df["sound_type"] = df["filename"].map(sound_type_from_filename)
        df["midi_note"] = df["filename"].map(midi_from_filename)
        df["frequency_hz"] = midi_to_hz(df["midi_note"])
        df.loc[df["sound_type"] == "noise", "midi_note"] = np.nan
        df.loc[df["sound_type"] == "noise", "frequency_hz"] = np.nan
        df["plot_freq_hz"] = df.apply(plot_freq_hz_from_row, axis=1)
        df["elev_error"] = df["pred_elev"] - df["true_elev"]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def filter_azimuth_zero(df, azim_col="true_azim"):
    return df[df[azim_col] == 0].copy()


def pearson_corr_stats(x, y):
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 2:
        return n, float("nan"), float("nan")
    r = x.corr(y, method="pearson")
    p = float("nan")
    if n >= 3 and not pd.isna(r):
        try:
            from scipy.stats import pearsonr

            _, p = pearsonr(x.to_numpy(), y.to_numpy())
        except ImportError:
            pass
    return n, r, p


def compute_error_frequency_correlations(df, error_col="elev_error"):
    rows = []
    for sound_type in SOUND_TYPES:
        for model_type in df["model_type"].unique():
            group = df[(df["sound_type"] == sound_type) & (df["model_type"] == model_type)]
            n, r, p = pearson_corr_stats(group["plot_freq_hz"], group[error_col])
            rows.append(
                {
                    "sound_type": sound_type,
                    "model_type": model_type,
                    "n": n,
                    "pearson_r": r,
                    "p_two_sided": p,
                }
            )
    return pd.DataFrame(rows)


def summarise_error_by_frequency(group, error_col="elev_error"):
    return (
        group.groupby("filename", sort=False)
        .agg(
            plot_freq_hz=("plot_freq_hz", "first"),
            mean_error=(error_col, "mean"),
            sem_error=(error_col, "sem"),
            n=(error_col, "count"),
        )
        .dropna(subset=["plot_freq_hz"])
        .sort_values("plot_freq_hz")
    )


def format_corr_annotation(n, r, p):
    if pd.isna(r):
        return f"n = {n}\nr = —"
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    sig = "*" if p < 0.05 else ""
    return f"n = {n}\nr = {r:.3f}{sig}\n{p_text}"


def set_log_freq_ticks(ax, freq_hz):
    freq_hz = np.asarray(freq_hz, dtype=float)
    freq_hz = freq_hz[np.isfinite(freq_hz) & (freq_hz > 0)]
    if len(freq_hz) == 0:
        return
    exp_min = int(np.floor(np.log10(freq_hz.min())))
    exp_max = int(np.ceil(np.log10(freq_hz.max())))
    ticks = [10.0**exp for exp in range(exp_min, exp_max + 1)]
    ax.set_xlim(10 ** (exp_min - 0.15), 10 ** (exp_max + 0.15))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))


def plot_error_panel(ax, subset, color, corr_stats, error_col="elev_error"):
    summary = summarise_error_by_frequency(subset, error_col=error_col)
    if summary.empty:
        ax.set_visible(False)
        return

    ax.errorbar(
        summary["plot_freq_hz"],
        summary["mean_error"],
        yerr=summary["sem_error"],
        fmt="o-",
        color=color,
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        markersize=5,
        alpha=0.9,
    )
    ax.set_xscale("log")
    set_log_freq_ticks(ax, summary["plot_freq_hz"].to_numpy())
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(
        0.03,
        0.97,
        format_corr_annotation(corr_stats["n"], corr_stats["pearson_r"], corr_stats["p_two_sided"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )


def plot_pratt_error_vs_frequency(
    df,
    plot_dir=PLOT_DIR,
    error_col="elev_error",
    prefix="regression_pratt_elevation_error_by_frequency",
    corr_df=None,
):
    os.makedirs(plot_dir, exist_ok=True)
    if corr_df is None:
        corr_df = compute_error_frequency_correlations(df, error_col=error_col)

    model_types = list(df["model_type"].unique())
    n_rows = len(SOUND_TYPES)
    n_cols = len(model_types)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    corr_lookup = {
        (row["sound_type"], row["model_type"]): row for _, row in corr_df.iterrows()
    }

    for row_idx, sound_type in enumerate(SOUND_TYPES):
        for col_idx, model_type in enumerate(model_types):
            ax = axes[row_idx, col_idx]
            subset = df[(df["sound_type"] == sound_type) & (df["model_type"] == model_type)]
            corr_stats = corr_lookup.get(
                (sound_type, model_type),
                {"n": 0, "pearson_r": np.nan, "p_two_sided": np.nan},
            )
            color = SOUND_TYPE_COLORS[sound_type]

            if subset.empty:
                ax.set_visible(False)
                continue

            plot_error_panel(ax, subset, color, corr_stats, error_col=error_col)
            title = SOUND_TYPE_LABELS[sound_type]
            if n_cols > 1:
                model_label = MODEL_TYPE_LABELS.get(model_type, model_type.replace("_", " ").title())
                title = f"{title} — {model_label}"
            ax.set_title(title, fontsize=10)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Frequency (Hz)")
            if col_idx == 0:
                ax.set_ylabel("Elevation error (pred − true, °)")

    fig.suptitle(
        "Pratt sounds: mean elevation error vs. frequency (± SEM, azimuth = 0°)",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()

    for ext in ("png", "svg"):
        out_path = os.path.join(plot_dir, f"{prefix}.{ext}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    df = load_pratt_predictions()
    print(f"Loaded {len(df):,} trials from {df['source_file'].nunique()} file(s)")
    print(f"Before azimuth filter: {len(df):,} trials")
    df = filter_azimuth_zero(df)
    print(f"After azimuth filter (true_azim = 0°): {len(df):,} trials\n")

    print("Trials by sound type:")
    print(df.groupby("sound_type").size())

    corr_df = compute_error_frequency_correlations(df)
    print("\nPearson correlation (elevation error vs. frequency):")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(corr_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_pratt_error_vs_frequency(df, corr_df=corr_df)
