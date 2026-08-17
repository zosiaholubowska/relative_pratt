"""
Load all regression prediction CSVs from regression_output/ into one dataframe.

Organisation:
  - condition: viola, viola_complex, harmoniccomplex, flute, artificial_sounds
  - model_type: test (tones/test run) or hrtf_flipped
  - frequency: derived from filename (MIDI note / f0 for tones; band index for artificial)
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

plt.rcParams["svg.fonttype"] = "none"

DIR = os.path.dirname(os.path.abspath(__file__))
REGRESSION_DIR = os.path.join(DIR, "regression_output")
PLOT_DIR = os.path.join(os.path.dirname(DIR), "plots")

CONDITIONS = (
    "viola",
    "viola_complex",
    "harmoniccomplex",
    "flute",
    "artificial_sounds",
)
TONE_CONDITIONS = {c for c in CONDITIONS if c != "artificial_sounds"}
MODEL_TYPES = ("test", "hrtf_flipped")

# Baseline run labels in filenames map to model_type "test".
MODEL_SUFFIX_TO_TYPE = {
    "tones": "test",
    "test": "test",
    "hrtf_flipped": "hrtf_flipped",
}

KHZ_BAND_ORDER = [
    "khz_band_lt_0.8_noise.wav",
    "khz_band_0.8_1.4_noise.wav",
    "khz_band_1.4_2.5_noise.wav",
    "khz_band_2.5_4.5_noise.wav",
    "khz_band_4.5_8_noise.wav",
    "khz_band_gt_8_noise.wav",
]

KHZ_BAND_LABELS = {
    "khz_band_lt_0.8_noise.wav": "<0.8 kHz",
    "khz_band_0.8_1.4_noise.wav": "0.8–1.4 kHz",
    "khz_band_1.4_2.5_noise.wav": "1.4–2.5 kHz",
    "khz_band_2.5_4.5_noise.wav": "2.5–4.5 kHz",
    "khz_band_4.5_8_noise.wav": "4.5–8 kHz",
    "khz_band_gt_8_noise.wav": ">8 kHz",
    "pink_noise.wav": "Pink noise",
}

# Ordinal frequency proxy for artificial-sound bands (ERB → kHz → pink).
ARTIFICIAL_FREQ_ORDER = [f"band_{i}_noise.wav" for i in range(10)] + KHZ_BAND_ORDER + [
    "pink_noise.wav"
]

CONDITION_LABELS = {
    "viola": "Viola",
    "viola_complex": "Viola complex",
    "harmoniccomplex": "Harmonic complex",
    "flute": "Flute",
    "artificial_sounds": "Artificial sounds",
}

MODEL_TYPE_LABELS = {
    "test": "Test",
    "hrtf_flipped": "HRTF flipped",
}

CONDITION_COLORS = {
    "flute": "#e22c1f",
    "harmoniccomplex": "#E5A09C",
    "viola": "#2e33a6",
    "viola_complex": "#9FA0CC",
    "artificial_sounds": "#6B8E9F",
}

KHZ_BAND_HZ_RANGES = {
    "khz_band_lt_0.8_noise.wav": (100.0, 800.0),
    "khz_band_0.8_1.4_noise.wav": (800.0, 1400.0),
    "khz_band_1.4_2.5_noise.wav": (1400.0, 2500.0),
    "khz_band_2.5_4.5_noise.wav": (2500.0, 4500.0),
    "khz_band_4.5_8_noise.wav": (4500.0, 8000.0),
    "khz_band_gt_8_noise.wav": (8000.0, 16000.0),
}

# Each row in the summary figure: (condition, artificial band subset or None for tones).
PLOT_ROWS = (
    ("viola", None),
    ("viola_complex", None),
    ("harmoniccomplex", None),
    ("flute", None),
    ("artificial_sounds", "erb"),
    ("artificial_sounds", "khz"),
)

ARTIFICIAL_BAND_SUBSET_LABELS = {
    "erb": "ERB bands",
    "khz": "kHz bands",
}

REGRESSION_FILENAME_RE = re.compile(
    r"^regression_predictions?_(?P<condition>.+)_(?P<suffix>tones|test|hrtf_flipped)\.csv$",
    re.IGNORECASE,
)

REQUIRED_COLS = {"filename", "true_azim", "true_elev", "pred_azim", "pred_elev"}


def midi_to_hz(midi):
    """Equal temperament: A4 (MIDI 69) = 440 Hz."""
    return 440.0 * (2.0 ** ((pd.Series(midi, dtype=float) - 69.0) / 12.0))


def note_from_fname(name):
    s = str(name).strip()
    m = re.search(r"stim_(\d+)_", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\.wav$", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return np.nan


def artificial_band_from_fname(name):
    s = str(name)
    m = re.search(r"band_(\d+)_noise", s)
    if m:
        return "erb", int(m.group(1))
    if s in KHZ_BAND_ORDER:
        return "khz", KHZ_BAND_ORDER.index(s)
    if "pink" in s.lower():
        return "pink", 0
    return "other", np.nan


def artificial_freq_order(name):
    try:
        return ARTIFICIAL_FREQ_ORDER.index(str(name))
    except ValueError:
        return np.nan


def build_artificial_band_center_hz(samplerate=44100):
    """Geometric-mean centre frequency (Hz) for each artificial band filename."""
    centers = {
        fn: float(np.sqrt(lo * hi)) for fn, (lo, hi) in KHZ_BAND_HZ_RANGES.items()
    }
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
            centers[f"band_{band}_noise.wav"] = float(
                np.sqrt(band_low_hz[band] * band_high_hz[band])
            )
    except ImportError:
        erb_centers = np.geomspace(150.0, 12000.0, 10)
        for band, center in enumerate(erb_centers):
            centers[f"band_{band}_noise.wav"] = float(center)
    centers["pink_noise.wav"] = float(np.sqrt(100.0 * (samplerate / 2)))
    return centers


ARTIFICIAL_BAND_CENTER_HZ = build_artificial_band_center_hz()


def plot_freq_hz_from_row(row):
    if row["condition"] == "artificial_sounds":
        return ARTIFICIAL_BAND_CENTER_HZ.get(str(row["filename"]), np.nan)
    return row["frequency_hz"]


def parse_regression_filename(path):
    base = os.path.basename(path)
    m = REGRESSION_FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Unexpected regression output filename: {base!r}")
    condition = m.group("condition").lower()
    suffix = m.group("suffix").lower()
    model_type = MODEL_SUFFIX_TO_TYPE[suffix]
    return condition, model_type, base


def add_frequency_columns(df):
    """Add columns describing the stimulus frequency encoded in filename."""
    out = df.copy()
    is_artificial = out["condition"] == "artificial_sounds"

    out["midi_note"] = out["filename"].map(note_from_fname)
    out.loc[is_artificial, "midi_note"] = np.nan
    out["frequency_hz"] = midi_to_hz(out["midi_note"])

    band_info = out["filename"].map(artificial_band_from_fname)
    out["band_kind"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["band_index"] = np.nan
    out.loc[is_artificial, "band_kind"] = [k for k, _ in band_info[is_artificial]]
    out.loc[is_artificial, "band_index"] = [i for _, i in band_info[is_artificial]]

    out["band_label"] = out["filename"].map(KHZ_BAND_LABELS)

    # Single x-axis for error-vs-frequency correlations.
    out["stimulus_freq_x"] = np.where(
        is_artificial,
        out["filename"].map(artificial_freq_order),
        out["frequency_hz"],
    )
    out["plot_freq_hz"] = out.apply(plot_freq_hz_from_row, axis=1)
    return out


def load_regression_csv(path):
    condition, model_type, source_file = parse_regression_filename(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["condition"] = condition
    df["model_type"] = model_type
    df["source_file"] = source_file
    df["elev_error"] = df["pred_elev"] - df["true_elev"]
    df["azim_error"] = df["pred_azim"] - df["true_azim"]
    return add_frequency_columns(df)


def filter_azimuth_zero(df, azim_col="true_azim"):
    """Keep only front trials (true azimuth = 0°)."""
    return df[df[azim_col] == 0].copy()


def load_all_regression_output(regression_dir=REGRESSION_DIR):
    paths = sorted(glob.glob(os.path.join(regression_dir, "regression_prediction*.csv")))
    if not paths:
        raise FileNotFoundError(f"No regression_prediction*.csv in {regression_dir}")

    frames = [load_regression_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)

    # Canonical column order: metadata first, then trial columns.
    meta_cols = ["condition", "model_type", "source_file"]
    freq_cols = [
        "midi_note",
        "frequency_hz",
        "band_kind",
        "band_index",
        "band_label",
        "stimulus_freq_x",
        "plot_freq_hz",
    ]
    trial_cols = [
        "filename",
        "true_azim",
        "true_elev",
        "pred_azim",
        "pred_elev",
        "elev_error",
        "azim_error",
    ]
    df = df[meta_cols + trial_cols + freq_cols]
    return df


def summarise_loaded_data(df):
    print(f"Loaded {len(df):,} trials from {df['source_file'].nunique()} files.\n")
    print("Trials by condition × model_type:")
    print(df.groupby(["condition", "model_type"], sort=False).size().unstack(fill_value=0))
    print("\nUnique stimuli (filename) per condition × model_type:")
    print(
        df.groupby(["condition", "model_type"])["filename"]
        .nunique()
        .unstack(fill_value=0)
    )


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


def freq_measure_label(condition, band_subset=None):
    if condition == "artificial_sounds":
        return ARTIFICIAL_BAND_SUBSET_LABELS.get(band_subset, "band order")
    return "f₀ (Hz)"


def subset_for_plot(df, condition, model_type, band_subset=None):
    subset = df[(df["condition"] == condition) & (df["model_type"] == model_type)]
    if band_subset == "erb":
        return subset[subset["band_kind"] == "erb"]
    if band_subset == "khz":
        return subset[subset["band_kind"].isin(["khz", "pink"])]
    return subset


def panel_title(condition, model_type, band_subset=None):
    condition_label = CONDITION_LABELS[condition]
    if band_subset is not None:
        condition_label = f"{condition_label} ({ARTIFICIAL_BAND_SUBSET_LABELS[band_subset]})"
    return f"{condition_label} — {MODEL_TYPE_LABELS[model_type]}"


def compute_error_frequency_correlations(df, error_col="elev_error"):
    """Pearson r between elevation error and stimulus frequency, per panel."""
    rows = []
    for condition, band_subset in PLOT_ROWS:
        for model_type in MODEL_TYPES:
            group = subset_for_plot(df, condition, model_type, band_subset)
            x = group["plot_freq_hz"]
            y = group[error_col]
            n, r, p = pearson_corr_stats(x, y)
            rows.append(
                {
                    "condition": condition,
                    "band_subset": band_subset or "",
                    "model_type": model_type,
                    "freq_measure": freq_measure_label(condition, band_subset),
                    "n": n,
                    "pearson_r": r,
                    "p_two_sided": p,
                }
            )
    return pd.DataFrame(rows)


def print_error_frequency_correlations(corr_df):
    print("\nPearson correlation (elevation error vs. frequency) by condition × model_type:")
    display_cols = ["condition", "band_subset", "model_type", "freq_measure", "n", "pearson_r", "p_two_sided"]
    display_cols = [c for c in display_cols if c in corr_df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(corr_df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def summarise_error_by_frequency(group, error_col="elev_error"):
    summary = (
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
    return summary


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
    pad_low = 10 ** (exp_min - 0.15)
    pad_high = 10 ** (exp_max + 0.15)
    ax.set_xlim(pad_low, pad_high)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
    )


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


def plot_error_vs_frequency_subplots(
    df,
    plot_dir=PLOT_DIR,
    error_col="elev_error",
    prefix="regression_elevation_error_by_frequency",
    corr_df=None,
):
    os.makedirs(plot_dir, exist_ok=True)
    if corr_df is None:
        corr_df = compute_error_frequency_correlations(df, error_col=error_col)

    corr_lookup = {
        (row["condition"], row["band_subset"], row["model_type"]): row
        for _, row in corr_df.iterrows()
    }

    n_rows = len(PLOT_ROWS)
    n_cols = len(MODEL_TYPES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for row_idx, (condition, band_subset) in enumerate(PLOT_ROWS):
        for col_idx, model_type in enumerate(MODEL_TYPES):
            ax = axes[row_idx, col_idx]
            subset = subset_for_plot(df, condition, model_type, band_subset)
            color = CONDITION_COLORS.get(condition, "#444444")
            corr_key = (condition, band_subset or "", model_type)
            corr_stats = corr_lookup.get(corr_key, {"n": 0, "pearson_r": np.nan, "p_two_sided": np.nan})

            if subset.empty:
                ax.set_visible(False)
                continue

            plot_error_panel(ax, subset, color, corr_stats, error_col=error_col)
            ax.set_title(panel_title(condition, model_type, band_subset), fontsize=10)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Frequency (Hz)")
            if col_idx == 0:
                ax.set_ylabel("Elevation error (pred − true, °)")

    fig.suptitle(
        "Mean elevation error vs. frequency (± SEM, azimuth = 0°)",
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
    all_pred = load_all_regression_output()
    print(f"Before azimuth filter: {len(all_pred):,} trials")
    all_pred = filter_azimuth_zero(all_pred)
    print(f"After azimuth filter (true_azim = 0°): {len(all_pred):,} trials\n")
    summarise_loaded_data(all_pred)
    corr_df = compute_error_frequency_correlations(all_pred)
    print_error_frequency_correlations(corr_df)
    plot_error_vs_frequency_subplots(all_pred, corr_df=corr_df)
