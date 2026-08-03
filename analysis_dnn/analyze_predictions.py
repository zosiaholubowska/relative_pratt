"""
Load regression_predictions_*_tones.csv in this folder.
X: note index from filename; Y: pred_elev - true_elev.

"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
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

def pearson_corr_stats(x, y):
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 2:
        return n, float("nan"), float("nan")
    r = x.corr(y, method="pearson")
    p = float("nan")
    if n >= 3 and not pandas.isna(r):
        try:
            from scipy.stats import pearsonr

            _, p = pearsonr(x.to_numpy(), y.to_numpy())
        except ImportError:
            pass
    return n, r, p


BEH_TO_DNN_CONDITION = {
    "complex": "harmoniccomplex",
    "flute": "flute",
    "viola": "viola",
    "viola_complex": "viola_complex",
}
SLOPE_MIN_TRIALS = 3
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 0


def fit_frequency_error_slope(x, y):
    valid = pandas.notna(x) & pandas.notna(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < SLOPE_MIN_TRIALS:
        return float("nan"), float("nan"), n
    try:
        from scipy.stats import linregress

        res = linregress(x.to_numpy(), y.to_numpy())
        return res.slope, res.intercept, n
    except ImportError:
        return float("nan"), float("nan"), n


def participant_frequency_slopes(df, condition_col, subject_col, x_col, y_col):
    rows = []
    for (condition, subject), group in df.groupby([condition_col, subject_col], sort=False):
        slope, intercept, n_trials = fit_frequency_error_slope(group[x_col], group[y_col])
        if pandas.isna(slope):
            continue
        rows.append(
            {
                "condition": condition,
                "subject": subject,
                "slope": slope,
                "intercept": intercept,
                "n_trials": n_trials,
            }
        )
    return pandas.DataFrame(rows)


def bootstrap_dnn_slopes(dnn_df, condition, n_slopes, x_col, y_col, rng):
    trials = dnn_df.loc[dnn_df["condition"] == condition, [x_col, y_col]].dropna()
    if len(trials) < SLOPE_MIN_TRIALS:
        return np.full(n_slopes, np.nan)

    slopes = np.empty(n_slopes, dtype=float)
    n_trials = len(trials)
    for i in range(n_slopes):
        idx = rng.integers(0, n_trials, size=n_trials)
        sample = trials.iloc[idx]
        slope, _, _ = fit_frequency_error_slope(sample[x_col], sample[y_col])
        slopes[i] = slope
    return slopes


def compare_slope_vectors(beh_slopes, dnn_slopes):
    from scipy.stats import pearsonr, ttest_rel

    mask = np.isfinite(beh_slopes) & np.isfinite(dnn_slopes)
    beh = beh_slopes[mask]
    dnn = dnn_slopes[mask]
    n = len(beh)
    if n < 2:
        return {
            "n": n,
            "pearson_r": float("nan"),
            "r_p_two_sided": float("nan"),
            "t_statistic": float("nan"),
            "t_p_two_sided": float("nan"),
            "mean_beh_slope": float(beh.mean()) if n else float("nan"),
            "mean_dnn_slope": float(dnn.mean()) if n else float("nan"),
            "mean_diff_beh_minus_dnn": float((beh - dnn).mean()) if n else float("nan"),
        }
    r, r_p = pearsonr(beh, dnn)
    t_stat, t_p = ttest_rel(beh, dnn)
    return {
        "n": n,
        "pearson_r": r,
        "r_p_two_sided": r_p,
        "t_statistic": t_stat,
        "t_p_two_sided": t_p,
        "mean_beh_slope": float(beh.mean()),
        "mean_dnn_slope": float(dnn.mean()),
        "mean_diff_beh_minus_dnn": float((beh - dnn).mean()),
    }


#paths = sorted(glob.glob(os.path.join(DIR, "analysis_dnn", "regression_predictions_*_tones.csv")))
paths = sorted(glob.glob(os.path.join(DIR, "analysis_dnn", "regression_predictions_*_bias.csv")))

if not paths:
    raise FileNotFoundError(f"No regression_predictions_*_tones.csv in {DIR}")

frames = []
for path in paths:
    base = os.path.basename(path)
    if base=='regression_predictions_artificial_sounds_bias.csv':
        continue
    m = re.match(r"regression_predictions_(.+)_(tones|bias)\.csv$", base, re.IGNORECASE)
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
all_pred = all_pred[all_pred.get("true_azim", 0) == 0]
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

# compare DNN and behavioural data (Pratt slope: frequency vs. elevation error)

beh_data = pandas.read_csv(os.path.join(DIR, "Results", "data.csv"))
beh_data = beh_data[~beh_data["subject"].str.contains("pilot", na=False)]
beh_data = beh_data[~beh_data["subject"].str.contains("00", na=False)]
beh_data = beh_data[beh_data["azimuth_ls"] == 0]
beh_data = beh_data[beh_data["condition"].isin(BEH_TO_DNN_CONDITION)]
beh_data["frequency_hz"] = beh_data["frequency"].round(0)

beh_slopes = participant_frequency_slopes(
    beh_data,
    condition_col="condition",
    subject_col="subject",
    x_col="frequency_hz",
    y_col="elevation_diff",
)
dnn_beh = all_pred.copy()

print("\nPratt slope comparison: participant slopes vs. bootstrapped DNN slopes")
print("Model: elevation error ~ frequency (error = response − loudspeaker for beh; pred − true for DNN)")
print(f"DNN subset: azimuth = 0° ({len(dnn_beh)} predictions)")
with pandas.option_context("display.max_columns", None, "display.width", 120):
    print("\nParticipant slopes per condition:")
    print(
        beh_slopes.groupby("condition")["slope"]
        .agg(n_subjects="count", mean="mean", std="std")
        .to_string()
    )

rng = np.random.default_rng(BOOTSTRAP_SEED)
slope_compare_rows = []
bootstrap_rows = []
for beh_cond, dnn_cond in BEH_TO_DNN_CONDITION.items():
    beh_cond_slopes = beh_slopes.loc[beh_slopes["condition"] == beh_cond, "slope"].to_numpy()
    n_subjects = len(beh_cond_slopes)
    if n_subjects == 0:
        continue

    dnn_slopes = bootstrap_dnn_slopes(
        dnn_beh,
        dnn_cond,
        n_subjects,
        x_col="frequency_hz",
        y_col="elev_error",
        rng=rng,
    )
    point = compare_slope_vectors(beh_cond_slopes, dnn_slopes)
    point.update({"condition": beh_cond, "dnn_condition": dnn_cond, "bootstrap": "point"})
    slope_compare_rows.append(point)

    rep_r = []
    rep_p = []
    for _ in range(BOOTSTRAP_REPLICATES):
        dnn_boot = bootstrap_dnn_slopes(
            dnn_beh,
            dnn_cond,
            n_subjects,
            x_col="frequency_hz",
            y_col="elev_error",
            rng=rng,
        )
        stats = compare_slope_vectors(beh_cond_slopes, dnn_boot)
        if np.isfinite(stats["pearson_r"]):
            rep_r.append(stats["pearson_r"])
        if np.isfinite(stats["t_p_two_sided"]):
            rep_p.append(stats["t_p_two_sided"])

    bootstrap_rows.append(
        {
            "condition": beh_cond,
            "n_subjects": n_subjects,
            "n_dnn_trials": int((dnn_beh["condition"] == dnn_cond).sum()),
            "r_median": float(np.median(rep_r)) if rep_r else float("nan"),
            "r_ci2.5": float(np.percentile(rep_r, 2.5)) if rep_r else float("nan"),
            "r_ci97.5": float(np.percentile(rep_r, 97.5)) if rep_r else float("nan"),
            "t_p_median": float(np.median(rep_p)) if rep_p else float("nan"),
        }
    )

slope_compare_df = pandas.DataFrame(slope_compare_rows)[
    [
        "condition",
        "n",
        "mean_beh_slope",
        "mean_dnn_slope",
        "mean_diff_beh_minus_dnn",
        "pearson_r",
        "r_p_two_sided",
        "t_statistic",
        "t_p_two_sided",
    ]
]
bootstrap_df = pandas.DataFrame(bootstrap_rows)

print("\nPoint comparison (one bootstrap draw of N DNN slopes per condition, N = n participants):")
with pandas.option_context("display.max_columns", None, "display.width", 140, "float_format", "{:.6f}".format):
    print(slope_compare_df.to_string(index=False))

print(f"\nBootstrap summary over {BOOTSTRAP_REPLICATES} replicates (correlation / paired t-test p-value):")
with pandas.option_context("display.max_columns", None, "display.width", 140, "float_format", "{:.6f}".format):
    print(bootstrap_df.to_string(index=False))
