import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(DIR, "Results")
REGRESSION_DIR = os.path.join(DIR, "analysis_dnn", "regression_output")

MODEL_COLUMNS = ("human", "basic", "flipped", "prob_bias")

# Load human data

human_data = pd.read_csv(os.path.join(RESULTS_DIR, "data.csv"))
human_data = human_data[~human_data["subject"].str.contains(r"pilot|00", regex=True, na=False)]
human_data = human_data.loc[human_data["condition"] == "flute"].copy()
human_data["filename"] = human_data["midi_note"].map(lambda midi: f"stim_{int(midi)}_flute.wav")
human_data["model"] = "human"

# Load regression data

file_suffixes = ("flute_basic", "flute_flipped", "flute_prob_bias")
selected_files = [
    os.path.join(REGRESSION_DIR, f"regression_predictions_{suffix}.csv") for suffix in file_suffixes
]
dfs = [
    pd.read_csv(path).assign(
        model=os.path.basename(path).removeprefix("regression_predictions_flute_").removesuffix(".csv")
    )
    for path in selected_files
]
df = pd.concat(dfs, ignore_index=True)
df["elevation_diff"] = df["pred_elev"] - df["true_elev"]

model_by_file = df.groupby(["filename", "model"], as_index=False).agg(
    elevation_diff=("elevation_diff", "mean")
)


# Compute similarity between human and model predictions

rows = []
subjects = human_data["subject"].unique()
for idx, subject in enumerate(subjects):
    print(f"Subject: {subject} [{idx + 1}/{len(subjects)}]")
    human_by_file = (
        human_data.loc[human_data["subject"] == subject]
        .groupby("filename", as_index=False)
        .agg(human_elev_diff=("elevation_diff", "mean"))
    )
    for model in df["model"].unique():
        model_files = (
            model_by_file.loc[model_by_file["model"] == model, ["filename", "elevation_diff"]]
            .rename(columns={"elevation_diff": "model_elev_diff"})
        )
        paired = human_by_file.merge(model_files, on="filename", how="inner").dropna(
            subset=["human_elev_diff", "model_elev_diff"]
        )
        n = len(paired)
        if n < 3:
            r, p = float("nan"), float("nan")
        else:
            r, p = pearsonr(paired["human_elev_diff"], paired["model_elev_diff"])
        rows.append(
            {
                "subject": subject,
                "model": model,
                "n_stimuli": n,
                "pearson_r": r,
                "p_value": p,
            }
        )

similarity = pd.DataFrame(rows)
out_csv = os.path.join(REGRESSION_DIR, "compare_regression_human_correlations.csv")
similarity.to_csv(out_csv, index=False)
print(similarity.to_string(index=False))
print(f"\nWrote {out_csv}")

# Plot similarity

sns.set_theme(style="whitegrid", palette="matplotlib")
sns.color_palette()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="model", y="pearson_r", data=similarity, hue="model", ci=95, capsize=.2, ax=ax)
ax.set_xlabel("Model")
ax.set_ylabel("Pearson R")
ax.set_title("Correlation between human and model predictions")

# Add thick black frame around plot edges
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2.5)

fig.savefig(os.path.join(DIR, "analysis_dnn", "plots", "compare_regression_human_correlations.png"))
plt.show()
plt.close(fig)