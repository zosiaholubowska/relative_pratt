import os
import librosa
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

DIR = os.getcwd()
STIM_ROOT = os.path.join(DIR, "stimuli")
STIM_DIRS = [
    "naturalsounds165",
    "tones/flute",
    "tones/viola",
    "tones/harmoniccomplex",
    "tones/viola_complex",
    "test_stimuli"
]

sounds = []
for stim_dir in STIM_DIRS:
    folder = os.path.join(STIM_ROOT, stim_dir)
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            sounds.append(
                {
                    "condition": os.path.basename(stim_dir),
                    "sound": fname,
                    "path": os.path.join(folder, fname),
                }
            )
print(f"Found {len(sounds)} sounds across {len(STIM_DIRS)} folders")

df = pd.DataFrame(columns=["condition", "sound", "centroid"])

# ------------------------------------------------------------
# Compute spectral centroid for each sound
# ------------------------------------------------------------

for idx, row in enumerate(sounds):
    print(f"Processing {idx+1} of {len(sounds)}: {row['condition']}/{row['sound']}")
    y, sr = librosa.load(row["path"])
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_midi = librosa.hz_to_midi(np.mean(centroid))
    print(f"{row['sound']}: {np.mean(centroid)}")
    df.loc[idx, "condition"] = row["condition"]
    df.loc[idx, "sound"] = row["sound"]
    df.loc[idx, "centroid"] = np.mean(centroid)
    df.loc[idx, "centroid_midi"] = centroid_midi


df.to_csv(os.path.join(DIR, "acoustic_analysis", "spectral_centroid_stimuli.csv"), index=False)


# ------------------------------------------------------------
# Plot the distribution of spectral centroid
# ------------------------------------------------------------

df = pd.read_csv(os.path.join(DIR, "acoustic_analysis", "spectral_centroid_stimuli.csv"))

conditions = [os.path.basename(d) for d in STIM_DIRS]
n_cond = len(conditions)
n_bins = 20
centroid_bins = np.linspace(df["centroid"].min(), df["centroid"].max(), n_bins + 1)
midi_bins = np.linspace(df["centroid_midi"].min(), df["centroid_midi"].max(), n_bins + 1)

fig, axs = plt.subplots(2, n_cond, figsize=(3.4 * n_cond, 7), squeeze=False)

for j, cond in enumerate(conditions):
    sub = df.loc[df["condition"] == cond]
    sns.histplot(sub["centroid"], bins=centroid_bins, ax=axs[0, j], color="steelblue")
    axs[0, j].set_title(cond)
    sns.histplot(sub["centroid_midi"], bins=midi_bins, ax=axs[1, j], color="steelblue")
    if j > 0:
        axs[0, j].set_ylabel("")
        axs[1, j].set_ylabel("")

for ax in axs[0, :]:
    ax.set_xlim(centroid_bins[0], centroid_bins[-1])
    ax.set_ylim(0, max(a.get_ylim()[1] for a in axs[0, :]))
for ax in axs[1, :]:
    ax.set_xlim(midi_bins[0], midi_bins[-1])
    ax.set_ylim(0, max(a.get_ylim()[1] for a in axs[1, :]))

axs[0, 0].set_ylabel("Count")
axs[1, 0].set_ylabel("Count")
for j in range(n_cond):
    axs[0, j].set_xlabel("Spectral Centroid (Hz)")
    axs[1, j].set_xlabel("Spectral Centroid (MIDI)")

plt.tight_layout()
plt.savefig(
    os.path.join(
        DIR,
        "acoustic_analysis",
        "plots",
        "spectral_centroid_training_stimuli.png",
    ),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Compare the natural sounds with the test stimuli

natural_sounds = df.loc[df["condition"] == "naturalsounds165"]
test_stimuli = df.loc[df["condition"] == "test_stimuli"]

print(f"Natural Sounds: {natural_sounds['centroid'].mean()}")
print(f"Test Stimuli: {test_stimuli['centroid'].mean()}")

plt.figure(figsize=(10, 5))
plt.hist(natural_sounds["centroid"], bins=centroid_bins, color="steelblue", label="Natural Sounds", alpha=0.5)
plt.hist(test_stimuli["centroid"], bins=centroid_bins, color="orange", label="Test Stimuli", alpha=0.5)
plt.legend()
plt.show()