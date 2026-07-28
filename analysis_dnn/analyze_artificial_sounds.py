import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns


PINK_NOISE_FILENAME = "pink_noise.wav"

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
}


def artificial_filename_order(names):
    def sort_key(name):
        s = str(name).lower()
        if "pink" in s:
            return (2, 0, s)
        m = re.search(r"^band_(\d+)_noise", s)
        if m:
            return (0, int(m.group(1)), s)
        if "khz_band" in s:
            try:
                return (1, KHZ_BAND_ORDER.index(name), s)
            except ValueError:
                return (1, 99, s)
        return (3, 0, s)

    return sorted(names, key=sort_key)


def erb_filename_order(names):
    erb = sorted(
        (n for n in names if re.search(r"^band_\d+_noise", str(n))),
        key=lambda n: int(re.search(r"band_(\d+)", str(n)).group(1)),
    )
    if PINK_NOISE_FILENAME in names:
        erb.append(PINK_NOISE_FILENAME)
    return erb


def khz_filename_order(names):
    khz = [n for n in KHZ_BAND_ORDER if n in names]
    if PINK_NOISE_FILENAME in names:
        khz.append(PINK_NOISE_FILENAME)
    return khz


def _format_freq_range_hz(low_hz, high_hz):
    if high_hz < 1000:
        return f"{round(low_hz)}–{round(high_hz)} Hz"
    return f"{low_hz / 1000:.1f}–{high_hz / 1000:.1f} kHz"


def build_artificial_filename_labels(samplerate=44100):
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
    labels = {
        f"band_{band}_noise.wav": _format_freq_range_hz(band_low_hz[band], band_high_hz[band])
        for band in range(len(band_low_hz))
    }
    labels.update(KHZ_BAND_LABELS)
    labels[PINK_NOISE_FILENAME] = "Pink noise"
    return labels


ARTIFICIAL_FILENAME_LABELS = build_artificial_filename_labels()


def artificial_filename_label(name):
    return ARTIFICIAL_FILENAME_LABELS.get(str(name), str(name))


def build_artificial_palette(filename_order, band_colors, pink_color="#444444"):
    palette = {fn: band_colors[i] for i, fn in enumerate(filename_order) if fn != PINK_NOISE_FILENAME}
    if PINK_NOISE_FILENAME in filename_order:
        palette[PINK_NOISE_FILENAME] = pink_color
    return palette


def plot_violin_box_scatter(
    ax,
    pred_df,
    filename_order,
    palette,
    value_col,
    *,
    show_zero_line=False,
    violin_offset=-0.18,
    strip_offset=0.18,
    violin_width=0.55,
    box_width=0.14,
    boxprops=None,
    medianprops=None,
    whiskerprops=None,
):
    positions = np.arange(len(filename_order))
    violin_data = [
        pred_df.loc[pred_df["filename"] == fn, value_col].to_numpy()
        for fn in filename_order
    ]

    violin_parts = ax.violinplot(
        violin_data,
        positions=positions + violin_offset,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor("#A9A9A9")
        body.set_edgecolor("none")
        body.set_alpha(0.75)
        verts = body.get_paths()[0].vertices
        center_x = verts[:, 0].mean()
        verts[:, 0] = np.minimum(verts[:, 0], center_x)

    for i, fn in enumerate(filename_order):
        data = pred_df.loc[pred_df["filename"] == fn, value_col]
        x = positions[i] + strip_offset + np.random.uniform(-0.08, 0.08, size=len(data))
        ax.scatter(
            x,
            data,
            color=palette[fn],
            alpha=0.55,
            s=14,
            edgecolor="none",
        )

    ax.boxplot(
        violin_data,
        positions=positions + 0.05,
        widths=box_width,
        manage_ticks=False,
        showfliers=False,
        showcaps=False,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=whiskerprops,
    )
    if show_zero_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [artificial_filename_label(fn) for fn in filename_order],
        rotation=45,
        ha="right",
    )
    ax.set_xlim(-0.6, len(filename_order) - 0.4)
    return positions


def plot_true_vs_pred(ax, pred_df, filename_order, palette, elev_min, elev_max):
    for fn in filename_order:
        subset = pred_df[pred_df["filename"] == fn]
        color = palette[fn]
        label = artificial_filename_label(fn)
        ax.scatter(
            subset["true_elev"],
            subset["pred_elev"],
            color=color,
            alpha=0.45,
            s=16,
            label=label,
        )
        sns.regplot(
            x="true_elev",
            y="pred_elev",
            data=subset,
            scatter=False,
            color=color,
            ax=ax,
        )
    ax.plot(
        [elev_min, elev_max],
        [elev_min, elev_max],
        color="black",
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="Identity",
    )
    ax.set_xlabel("True elevation")
    ax.set_ylabel("Predicted elevation")


plt.rcParams["svg.fonttype"] = "none"
DIR = os.getcwd()
PLOT_DIR = os.path.join(DIR, "plots")

pred_path = os.path.join(DIR, "analysis_dnn", "regression_predictions_artificial_sounds.csv")
pred_df = pandas.read_csv(pred_path)
need = {"filename", "true_elev", "pred_elev"}
if need - set(pred_df.columns):
    raise ValueError(f"{pred_path}: missing columns {need - set(pred_df.columns)}")
pred_df["elev_error"] = pred_df["pred_elev"] - pred_df["true_elev"]

filename_order = artificial_filename_order(pred_df["filename"].unique())
erb_order = erb_filename_order(pred_df["filename"].unique())
khz_order = khz_filename_order(pred_df["filename"].unique())
erb_colors = sns.color_palette("tab10", n_colors=10)
khz_colors = sns.color_palette("Set2", n_colors=len(KHZ_BAND_ORDER))
ERB_PALETTE = build_artificial_palette(erb_order, erb_colors)
KHZ_PALETTE = build_artificial_palette(khz_order, khz_colors)
ARTIFICIAL_PALETTE = {**ERB_PALETTE, **KHZ_PALETTE}

boxprops = {"edgecolor": "black", "facecolor": "white", "linewidth": 1.0}
medianprops = {"color": "black", "linewidth": 1.5}
whiskerprops = {"color": "black", "linewidth": 1.0}

bias_summary = (
    pred_df.groupby("filename", sort=False)["elev_error"]
    .agg(["count", "mean", "std"])
    .reindex(filename_order)
)
print("\nArtificial sounds: elevation error (pred − true) by filename:")
with pandas.option_context("display.max_columns", None, "display.width", 120):
    print(bias_summary.to_string())

# Plot 1: half-violin + jittered scatter + boxplot (error by filename)
fig_art1, (ax_khz1, ax_erb1) = plt.subplots(2, 1, figsize=(12, 10), sharey=True)

plot_violin_box_scatter(
    ax_khz1,
    pred_df,
    khz_order,
    KHZ_PALETTE,
    "elev_error",
    show_zero_line=True,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
)
ax_khz1.set_title("kHz bands")
ax_khz1.set_ylabel("Predicted elevation − true elevation")

plot_violin_box_scatter(
    ax_erb1,
    pred_df,
    erb_order,
    ERB_PALETTE,
    "elev_error",
    show_zero_line=True,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
)
ax_erb1.set_title("ERB bands (0–9)")
ax_erb1.set_xlabel("Artificial sound")
ax_erb1.set_ylabel("Predicted elevation − true elevation")

fig_art1.suptitle("DNN elevation error by artificial sound", y=1.01)
plt.tight_layout()
out_art1_png = os.path.join(PLOT_DIR, "dnn_artificial_elevation_error_by_filename.png")
out_art1_svg = os.path.join(PLOT_DIR, "dnn_artificial_elevation_error_by_filename.svg")
plt.savefig(out_art1_png, dpi=200, bbox_inches="tight")
plt.savefig(out_art1_svg, bbox_inches="tight")
plt.close()
print(f"Wrote {out_art1_png}")
print(f"Wrote {out_art1_svg}")

# Plot 2: true vs. predicted elevation, one colour per filename
fig_art2, (ax_khz2, ax_erb2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
elev_min = min(pred_df["true_elev"].min(), pred_df["pred_elev"].min())
elev_max = max(pred_df["true_elev"].max(), pred_df["pred_elev"].max())

plot_true_vs_pred(ax_khz2, pred_df, khz_order, KHZ_PALETTE, elev_min, elev_max)
ax_khz2.set_title("kHz bands")
ax_khz2.legend(title="Sound", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)

plot_true_vs_pred(ax_erb2, pred_df, erb_order, ERB_PALETTE, elev_min, elev_max)
ax_erb2.set_title("ERB bands (0–9)")
ax_erb2.legend(title="Sound", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)

fig_art2.suptitle("DNN elevation: true vs. predicted (artificial sounds)", y=1.02)
plt.tight_layout()
out_art2_png = os.path.join(PLOT_DIR, "dnn_artificial_true_vs_pred_elevation.png")
out_art2_svg = os.path.join(PLOT_DIR, "dnn_artificial_true_vs_pred_elevation.svg")
plt.savefig(out_art2_png, dpi=200, bbox_inches="tight")
plt.savefig(out_art2_svg, bbox_inches="tight")
plt.close()
print(f"Wrote {out_art2_png}")
print(f"Wrote {out_art2_svg}")

pred_summary = (
    pred_df.groupby("filename", sort=False)["pred_elev"]
    .agg(["count", "mean", "std", "min", "max"])
    .reindex(filename_order)
)
print("\nArtificial sounds: predicted elevation by filename:")
with pandas.option_context("display.max_columns", None, "display.width", 120):
    print(pred_summary.to_string())

# Plot 3: half-violin + jittered scatter + boxplot (pred_elev by filename)
fig_art3, (ax_khz3, ax_erb3) = plt.subplots(2, 1, figsize=(12, 10), sharey=True)

plot_violin_box_scatter(
    ax_khz3,
    pred_df,
    khz_order,
    KHZ_PALETTE,
    "pred_elev",
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
)
ax_khz3.set_title("kHz bands")
ax_khz3.set_ylabel("Predicted elevation")

plot_violin_box_scatter(
    ax_erb3,
    pred_df,
    erb_order,
    ERB_PALETTE,
    "pred_elev",
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
)
ax_erb3.set_title("ERB bands (0–9)")
ax_erb3.set_xlabel("Artificial sound")
ax_erb3.set_ylabel("Predicted elevation")

fig_art3.suptitle("DNN predicted elevation by artificial sound", y=1.01)
plt.tight_layout()
out_art3_png = os.path.join(PLOT_DIR, "dnn_artificial_pred_elevation_by_filename.png")
out_art3_svg = os.path.join(PLOT_DIR, "dnn_artificial_pred_elevation_by_filename.svg")
plt.savefig(out_art3_png, dpi=200, bbox_inches="tight")
plt.savefig(out_art3_svg, bbox_inches="tight")
plt.close()
print(f"Wrote {out_art3_png}")
print(f"Wrote {out_art3_svg}")

