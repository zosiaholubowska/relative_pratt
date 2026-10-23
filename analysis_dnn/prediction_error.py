import os

import matplotlib.pyplot as plt
import pandas
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

DIR = os.getcwd()
REGRESSION_DIR = os.path.join(DIR, "analysis_dnn", "regression_output")
PLOT_DIR = os.path.join(DIR, "analysis_dnn", "plots")

# find files

files = [f for f in os.listdir(REGRESSION_DIR) if f.startswith("regression_predictions_")]
print(f"Found {len(files)} files")

def calculate_prediction_error(df):
    error_measures = []
    for filename in df["filename"].unique():
        df_filename = df[df["filename"] == filename]
        error_sd = df_filename["elevation_diff"].std()
        error_amplitude = df_filename["elevation_diff"].max() - df_filename["elevation_diff"].min()
        error_mean = df_filename["elevation_diff"].mean()
        error_measures.append({
            "filename": filename,
            "error_sd": error_sd,
            "error_amplitude": error_amplitude,
            "error_mean": error_mean
        })
    return pandas.DataFrame(error_measures)

def plot_prediction_error(error_measures, file):
    data = error_measures.copy()
    if "centroid_midi" not in data.columns:
        data["centroid_midi"] = data["filename"].str.extract(r"stim_(\d+)_", expand=False).astype(float)
    data = data.dropna(subset=["centroid_midi"]).sort_values("centroid_midi")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    sns.scatterplot(data=data, x="centroid_midi", y="error_sd", ax=axes[0], color="steelblue", s=36)
    axes[0].axhline(data["error_sd"].mean(), color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("MIDI note")
    axes[0].set_ylabel("Error SD (°)")

    sns.scatterplot(data=data, x="centroid_midi", y="error_amplitude", ax=axes[1], color="steelblue", s=36)
    axes[1].axhline(data["error_amplitude"].mean(), color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("MIDI note")
    axes[1].set_ylabel("Error amplitude (°)")

    fig.tight_layout()

    out_dir = os.path.join(PLOT_DIR, "error_measures")
    os.makedirs(out_dir, exist_ok=True)
    stem = str(file).removeprefix("regression_predictions_").removesuffix(".csv")
    fig.savefig(os.path.join(out_dir, f"{stem}.svg"))
    plt.close(fig)

all_error_measures = []

for file in files:
    file_path = os.path.join(REGRESSION_DIR, file)
    print(f"Loading {file_path}")
    df = pandas.read_csv(file_path)
    df["elevation_diff"] = df["pred_elev"] - df["true_elev"]

    centroids = pandas.read_csv(os.path.join(DIR, "acoustic_analysis", "spectral_centroid_stimuli.csv")).set_index("sound")[["centroid", "centroid_midi"]]
    df["centroid"] = df["filename"].map(centroids["centroid"])
    df["centroid_midi"] = df["filename"].map(centroids["centroid_midi"])

    error_measures = calculate_prediction_error(df)
    error_measures["model"] = file.removeprefix("regression_predictions_").removesuffix(".csv")

    all_error_measures.append(error_measures)

all_error_measures = pandas.concat(all_error_measures, ignore_index=True)
all_error_measures.to_csv(os.path.join(REGRESSION_DIR, "all_error_measures.csv"), index=False)

summary = all_error_measures.groupby("model").agg({
    "error_sd": ["mean", "std"],
    "error_amplitude": ["mean", "std"],
    "error_mean": ["mean", "std"]
})

print("Prediction Error Summary by Model:")
print(summary)

# plot the summary

SOUND_SOURCES = ("naturalsounds", "harmoniccomplex", "flute", "viola")
set1_colors = plt.get_cmap("Set1").colors
SOURCE_COLORS = {
    "flute": set1_colors[0],             # red
    "harmoniccomplex": set1_colors[1],   # blue
    "viola": set1_colors[2],             # green
    "naturalsounds": set1_colors[4],     # orange
}
SOURCE_LABELS = {
    "flute": "Flute",
    "viola": "Viola",
    "naturalsounds": "Natural sounds",
    "harmoniccomplex": "Harmonic sound",
}
MODEL_TYPE_ORDER = ["basic", "flipped", "prob_bias", "basic_tones", "prob_bias_tones"]


def parse_model_name(model):
    for source in SOUND_SOURCES:
        prefix = f"{source}_"
        if model.startswith(prefix):
            return source, model[len(prefix) :]
    return None, model


plot_df = all_error_measures.copy()
parsed = plot_df["model"].map(parse_model_name)
plot_df["sound_source"] = parsed.map(lambda x: x[0])
plot_df["model_type"] = parsed.map(lambda x: x[1])
plot_df = plot_df.dropna(subset=["sound_source"])

model_types = [m for m in MODEL_TYPE_ORDER if m in plot_df["model_type"].unique()]
sources = [s for s in SOUND_SOURCES if s in plot_df["sound_source"].unique()]

bar_stats = (
    plot_df.groupby(["model_type", "sound_source"], as_index=False)
    .agg(error_sd=("error_sd", "mean"), error_amplitude=("error_amplitude", "mean"))
)
bar_err = (
    plot_df.groupby(["model_type", "sound_source"], as_index=False)
    .agg(error_sd=("error_sd", "std"), error_amplitude=("error_amplitude", "std"))
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
metrics = [("error_sd", "Error SD (°)"), ("error_amplitude", "Error amplitude (°)")]

n_sources = len(sources)
bar_width = 0.8 / max(n_sources, 1)
x_base = range(len(model_types))

for ax, (metric, ylabel) in zip(axes, metrics):
    for i, source in enumerate(sources):
        means = bar_stats[bar_stats["sound_source"] == source].set_index("model_type")
        errs = bar_err[bar_err["sound_source"] == source].set_index("model_type")
        x = [j + (i - (n_sources - 1) / 2) * bar_width for j in x_base]
        y = [means.loc[mt, metric] if mt in means.index else float("nan") for mt in model_types]
        yerr = [errs.loc[mt, metric] if mt in errs.index else float("nan") for mt in model_types]
        ax.bar(
            x,
            y,
            width=bar_width,
            yerr=yerr,
            capsize=3,
            color=SOURCE_COLORS[source],
            label=SOURCE_LABELS[source],
            error_kw={"linewidth": 1, "ecolor": "black"},
        )
    ax.set_xticks(list(x_base))
    ax.set_xticklabels(model_types, rotation=30, ha="right")
    ax.set_xlabel("Model type")
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.5)

axes[1].legend(frameon=True, loc="lower right")
fig.tight_layout()
os.makedirs(PLOT_DIR, exist_ok=True)
fig.savefig(os.path.join(PLOT_DIR, "prediction_error_by_model_type.png"), dpi=300)
plt.close(fig)