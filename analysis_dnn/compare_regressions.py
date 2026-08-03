"""
Compare two regression runs (e.g. bias vs bias_2) on the same stimuli/positions,
and contrast run-to-run prediction spread with bias vs a baseline run on pred_elev.

Rows are matched by position in the CSV (same export order). Trial keys
(filename, true_azim, true_elev) are checked row-wise but are not unique
within a file, so key-based merges would inflate row counts.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

DIR = os.path.dirname(os.path.abspath(__file__))
REGRESSION_DIR = os.path.join(DIR, "regression_output")
PLOT_DIR = os.path.join(os.path.dirname(DIR), "plots")

KEY_COLS = ("filename", "true_azim", "true_elev")
ELEV_COL = "pred_elev"

# (condition slug, run labels: bias round 1, bias round 2, baseline e.g. tones or test)
COMPARISONS = (
    ("viola_complex", ("bias", "bias_2", "tones")),
    ("artificial_sounds", ("bias", "bias_2", "test")),
)

PLOT_COLOR = {
    "viola_complex": "#9FA0CC",
    "artificial_sounds": "#6B8E9F",
}

KHZ_BAND_ORDER = [
    "khz_band_lt_0.8_noise.wav",
    "khz_band_0.8_1.4_noise.wav",
    "khz_band_1.4_2.5_noise.wav",
    "khz_band_2.5_4.5_noise.wav",
    "khz_band_4.5_8_noise.wav",
    "khz_band_gt_8_noise.wav",
]


def midi_to_hz(midi):
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


def elev_error_vs_x_stats(df, x_col: str) -> dict:
    from scipy.stats import linregress, pearsonr

    x = df[x_col]
    y = df["elev_error"]
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 2:
        return {"n": n, "pearson_r": float("nan"), "p_two_sided": float("nan"), "slope": float("nan")}
    r, p = pearsonr(x.to_numpy(), y.to_numpy())
    slope, intercept, _, _, _ = linregress(x.to_numpy(), y.to_numpy())
    return {
        "n": n,
        "pearson_r": float(r),
        "p_two_sided": float(p),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def prepare_viola_f0_frame(condition: str, run: str) -> pandas.DataFrame:
    df = load_run(condition, run)
    if "true_azim" in df.columns:
        df = df[df["true_azim"] == 0].copy()
    df["elev_error"] = df[ELEV_COL] - df["true_elev"]
    df["midi_note"] = df["filename"].map(note_from_fname)
    df["frequency_hz"] = midi_to_hz(df["midi_note"])
    df["run"] = run
    return df


def artificial_band_kind_index(filename):
    s = str(filename)
    m = re.search(r"band_(\d+)_noise", s)
    if m:
        return "erb", int(m.group(1))
    if s in KHZ_BAND_ORDER:
        return "khz", KHZ_BAND_ORDER.index(s)
    if "pink" in s.lower():
        return "pink", 0
    return "other", float("nan")


def prepare_artificial_band_frame(condition: str, run: str) -> pandas.DataFrame:
    df = load_run(condition, run)
    df["elev_error"] = df[ELEV_COL] - df["true_elev"]
    kind_idx = df["filename"].map(artificial_band_kind_index)
    df["band_kind"] = [k for k, _ in kind_idx]
    df["band_index"] = [i for _, i in kind_idx]
    df["run"] = run
    return df


def compare_f0_relation(condition: str, run_bias: str, run_baseline: str) -> None:
    """Pratt-style check: does bias vs baseline change elev_error vs f₀?"""
    color = PLOT_COLOR.get(condition, "#888888")
    bias_df = prepare_viola_f0_frame(condition, run_bias)
    base_df = prepare_viola_f0_frame(condition, run_baseline)

    rows = []
    for run, df in ((run_bias, bias_df), (run_baseline, base_df)):
        for x_col, x_label in (("midi_note", "MIDI note"), ("frequency_hz", "f₀ (Hz)")):
            stats = elev_error_vs_x_stats(df, x_col)
            rows.append({"run": run, "predictor": x_label, **stats})
    stats_df = pandas.DataFrame(rows)

    prof_bias = bias_df.groupby("midi_note", sort=True)["elev_error"].mean()
    prof_base = base_df.groupby("midi_note", sort=True)["elev_error"].mean()
    notes = prof_bias.index.intersection(prof_base.index)
    n_prof, r_prof, p_prof = pearson_corr_stats(prof_bias[notes], prof_base[notes])

    print(f"\n--- {condition}: elevation error vs f₀ (true_azim = 0°) ---")
    print("Trial-level Pearson r and linear slope (elev_error ~ predictor):")
    with pandas.option_context("display.max_columns", None, "display.width", 120):
        print(stats_df.to_string(index=False))
    print(
        f"Mean error-by-note profile ({run_bias} vs {run_baseline}): "
        f"n_notes={n_prof}, r={r_prof:.4f}, p={p_prof:.4g}"
    )

    os.makedirs(PLOT_DIR, exist_ok=True)
    prefix = f"dnn_{condition}_regression_compare"
    fig, ax = plt.subplots(figsize=(8, 5))
    notes_sorted = sorted(notes)
    ax.plot(
        notes_sorted,
        prof_bias.reindex(notes_sorted),
        "o-",
        label=run_bias,
        color=color,
        markersize=5,
    )
    ax.plot(
        notes_sorted,
        prof_base.reindex(notes_sorted),
        "s--",
        label=run_baseline,
        color="#444444",
        markersize=5,
        alpha=0.85,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("MIDI note")
    ax.set_ylabel("Mean elevation error (pred − true, °)")
    ax.set_title(f"{condition}: mean error vs note ({run_bias} vs {run_baseline})")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(PLOT_DIR, f"{prefix}_error_vs_midi_profile.{ext}"))
    plt.close(fig)


def compare_band_relation(condition: str, run_bias: str, run_baseline: str) -> None:
    """Does bias vs baseline change elev_error vs ERB / kHz band index?"""
    color = PLOT_COLOR.get(condition, "#888888")
    bias_df = prepare_artificial_band_frame(condition, run_bias)
    base_df = prepare_artificial_band_frame(condition, run_baseline)

    rows = []
    for run, df in ((run_bias, bias_df), (run_baseline, base_df)):
        for kind, label in (("erb", "ERB band index"), ("khz", "kHz band index")):
            sub = df[df["band_kind"] == kind]
            stats = elev_error_vs_x_stats(sub, "band_index")
            rows.append({"run": run, "band_set": label, **stats})
    stats_df = pandas.DataFrame(rows)

    prof_bias = bias_df.groupby("filename", sort=False)["elev_error"].mean()
    prof_base = base_df.groupby("filename", sort=False)["elev_error"].mean()
    common = prof_bias.index.intersection(prof_base.index)
    n_prof, r_prof, p_prof = pearson_corr_stats(prof_bias[common], prof_base[common])

    print(f"\n--- {condition}: elevation error vs band ---")
    print("Trial-level Pearson r and slope (elev_error ~ band index; ERB and kHz subsets):")
    with pandas.option_context("display.max_columns", None, "display.width", 120):
        print(stats_df.to_string(index=False))
    print(
        f"Mean error-by-stimulus profile ({run_bias} vs {run_baseline}): "
        f"n_sounds={n_prof}, r={r_prof:.4f}, p={p_prof:.4g}"
    )

    os.makedirs(PLOT_DIR, exist_ok=True)
    prefix = f"dnn_{condition}_regression_compare"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind, title in zip(
        axes,
        ("erb", "khz"),
        ("ERB bands (0–9)", "kHz noise bands"),
    ):
        for run, df, ls in (
            (run_bias, bias_df, "-"),
            (run_baseline, base_df, "--"),
        ):
            means = (
                df[df["band_kind"] == kind]
                .groupby("band_index", sort=True)["elev_error"]
                .mean()
            )
            ax.plot(
                means.index,
                means.values,
                "o" + ls,
                label=run,
                color=color if run == run_bias else "#444444",
                markersize=5,
                alpha=0.9 if run == run_bias else 0.85,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Band index")
        ax.set_title(title)
    axes[0].set_ylabel("Mean elevation error (pred − true, °)")
    axes[1].legend()
    fig.suptitle(f"{condition}: mean error vs band ({run_bias} vs {run_baseline})")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(PLOT_DIR, f"{prefix}_error_vs_band_profile.{ext}"))
    plt.close(fig)


def load_run(condition: str, run_label: str) -> pandas.DataFrame:
    path = os.path.join(
        REGRESSION_DIR, f"regression_predictions_{condition}_{run_label}.csv"
    )
    df = pandas.read_csv(path)
    need = set(KEY_COLS) | {ELEV_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def align_runs(runs: tuple[str, ...], frames: dict[str, pandas.DataFrame]) -> pandas.DataFrame:
    ref = runs[0]
    n = len(frames[ref])
    for run, df in frames.items():
        if len(df) != n:
            raise ValueError(f"Row count mismatch: {ref} has {n}, {run} has {len(df)}")
        if not (df[list(KEY_COLS)].values == frames[ref][list(KEY_COLS)].values).all():
            raise ValueError(f"Row alignment mismatch between {ref} and {run}")

    out = frames[ref][list(KEY_COLS)].copy()
    for run in runs:
        out[f"{ELEV_COL}_{run}"] = frames[run][ELEV_COL].to_numpy()
    return out


def pearson_corr_stats(x, y):
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 2:
        return n, float("nan"), float("nan")
    r = x.corr(y, method="pearson")
    p = float("nan")
    if n >= 3 and not pandas.isna(r):
        from scipy.stats import pearsonr

        _, p = pearsonr(x.to_numpy(), y.to_numpy())
    return n, r, p


def diff_summary(abs_diff: pandas.Series) -> dict:
    d = abs_diff.dropna()
    return {
        "n": int(len(d)),
        "mean_abs": float(d.mean()),
        "median_abs": float(d.median()),
        "rmse": float(np.sqrt((d**2).mean())),
        "max_abs": float(d.max()),
    }


def compare_condition(condition: str, runs: tuple[str, str, str]) -> None:
    run_a, run_b, run_base = runs
    frames = {run: load_run(condition, run) for run in runs}
    merged = align_runs(runs, frames)
    elev_cols = {run: f"{ELEV_COL}_{run}" for run in runs}
    color = PLOT_COLOR.get(condition, "#888888")

    merged["abs_diff_bias_bias2"] = (
        merged[elev_cols[run_a]] - merged[elev_cols[run_b]]
    ).abs()
    merged["abs_diff_bias_baseline"] = (
        merged[elev_cols[run_a]] - merged[elev_cols[run_base]]
    ).abs()

    label_bias2 = f"bias vs {run_b}"
    label_baseline = f"bias vs {run_base}"

    if np.allclose(
        merged[elev_cols[run_a]], merged[elev_cols[run_b]], rtol=0, atol=1e-9
    ):
        print(
            f"[{condition}] Note: pred_elev in {run_a} and {run_b} are identical "
            "row-wise (same regression output on disk).\n"
        )

    pairs = [
        (run_a, run_b),
        (run_a, run_base),
        (run_b, run_base),
    ]
    corr_rows = []
    for a, b in pairs:
        n, r, p = pearson_corr_stats(merged[elev_cols[a]], merged[elev_cols[b]])
        corr_rows.append(
            {
                "pair": f"{a} vs {b}",
                "n": n,
                "pearson_r_pred_elev": r,
                "p_two_sided": p,
            }
        )
    corr_df = pandas.DataFrame(corr_rows)

    diff_rows = [
        {**{"comparison": label_bias2}, **diff_summary(merged["abs_diff_bias_bias2"])},
        {
            **{"comparison": label_baseline},
            **diff_summary(merged["abs_diff_bias_baseline"]),
        },
    ]
    diff_df = pandas.DataFrame(diff_rows)

    from scipy.stats import ttest_rel

    paired = ttest_rel(
        merged["abs_diff_bias_bias2"].to_numpy(),
        merged["abs_diff_bias_baseline"].to_numpy(),
    )

    print(f"Condition: {condition} ({len(merged)} rows, row-aligned)")
    print("\nPearson correlation of predicted elevation:")
    with pandas.option_context("display.max_columns", None, "display.width", 120):
        print(corr_df.to_string(index=False))

    print("\nAbsolute prediction difference |Δ pred_elev| (degrees):")
    with pandas.option_context("display.max_columns", None, "display.width", 120):
        print(diff_df.to_string(index=False))

    print(
        f"\nPaired t-test |{label_bias2}| vs |{label_baseline}|: "
        f"t={paired.statistic:.4f}, p={paired.pvalue:.6g} "
        f"(mean |bias−{run_b}|={merged['abs_diff_bias_bias2'].mean():.3f}, "
        f"mean |bias−{run_base}|={merged['abs_diff_bias_baseline'].mean():.3f})"
    )
    print()

    os.makedirs(PLOT_DIR, exist_ok=True)
    prefix = f"dnn_{condition}_regression_compare"

    fig, ax = plt.subplots(figsize=(6, 6))
    x = merged[elev_cols[run_a]]
    y = merged[elev_cols[run_b]]
    ax.scatter(x, y, s=8, alpha=0.25, color=color, edgecolors="none")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.7)
    _, r, _ = pearson_corr_stats(x, y)
    ax.set_xlabel(f"Predicted elevation — {run_a} (°)")
    ax.set_ylabel(f"Predicted elevation — {run_b} (°)")
    ax.set_title(f"{condition}: {run_a} vs {run_b} (r = {r:.3f})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(PLOT_DIR, f"{prefix}_bias_vs_bias2_scatter.{ext}"))
    plt.close(fig)

    plot_long = pandas.concat(
        [
            merged[["abs_diff_bias_bias2"]]
            .rename(columns={"abs_diff_bias_bias2": "abs_diff"})
            .assign(contrast=label_bias2),
            merged[["abs_diff_bias_baseline"]]
            .rename(columns={"abs_diff_bias_baseline": "abs_diff"})
            .assign(contrast=label_baseline),
        ],
        ignore_index=True,
    )
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=plot_long,
        x="contrast",
        y="abs_diff",
        order=[label_bias2, label_baseline],
        color=color,
        inner="box",
        ax=ax2,
    )
    ax2.set_ylabel("|Δ predicted elevation| (°)")
    ax2.set_xlabel("")
    ax2.set_title(f"{condition}: run-to-run vs bias–baseline spread")
    fig2.tight_layout()
    for ext in ("png", "svg"):
        fig2.savefig(os.path.join(PLOT_DIR, f"{prefix}_abs_pred_diff.{ext}"))
    plt.close(fig2)

    fig3, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, run in zip(axes, runs):
        pred = merged[elev_cols[run]]
        true = merged["true_elev"]
        ax.scatter(true, pred, s=6, alpha=0.2, color=color, edgecolors="none")
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.7)
        ax.set_title(run)
        ax.set_xlabel("True elevation (°)")
    axes[0].set_ylabel("Predicted elevation (°)")
    fig3.suptitle(f"{condition}: true vs predicted elevation by run")
    fig3.tight_layout()
    for ext in ("png", "svg"):
        fig3.savefig(os.path.join(PLOT_DIR, f"{prefix}_true_vs_pred_by_run.{ext}"))
    plt.close(fig3)


def main():
    for condition, runs in COMPARISONS:
        compare_condition(condition, runs)
        run_bias, _, run_baseline = runs
        if condition == "viola_complex":
            compare_f0_relation(condition, run_bias, run_baseline)
        elif condition == "artificial_sounds":
            compare_band_relation(condition, run_bias, run_baseline)


if __name__ == "__main__":
    main()
