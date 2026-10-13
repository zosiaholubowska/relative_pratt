import os
import librosa
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

DIR = os.getcwd()
STIM_DIR = os.path.join(DIR, "stimuli", "naturalsounds165")

sounds = [f for f in os.listdir(STIM_DIR) if f.endswith('.wav')]
print(f"Found {len(sounds)} sounds in {STIM_DIR}")
print("Should be 165.")

df = pd.DataFrame(columns=["sound", "centroid"])

# ------------------------------------------------------------
# Compute spectral centroid for each sound
# ------------------------------------------------------------

for idx, sound in enumerate(sounds):
    print(f"Processing {idx+1} of {len(sounds)}: {sound}")
    y, sr = librosa.load(os.path.join(STIM_DIR, sound))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_midi = librosa.hz_to_midi(np.mean(centroid))
    print(f"{sound}: {np.mean(centroid)}")
    df.loc[idx, "sound"] = sound
    df.loc[idx, "centroid"] = np.mean(centroid)
    df.loc[idx, "centroid_midi"] = centroid_midi


df.to_csv(os.path.join(DIR, "acoustic_analysis", "spectral_centroid_training_stimuli.csv"), index=False)

# ------------------------------------------------------------
# Plot the distribution of spectral centroid
# ------------------------------------------------------------

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# spectral centroid (Hz)
sns.histplot(df["centroid"], bins=20, ax=axs[0])
axs[0].set_xlabel("Spectral Centroid (Hz)")
axs[0].set_ylabel("Count")
axs[0].set_title("Spectral Centroid (Hz) Distribution")

# spectral centroid (MIDI)
sns.histplot(df["centroid_midi"], bins=20, ax=axs[1])
axs[1].set_xlabel("Spectral Centroid (MIDI)")
axs[1].set_ylabel("Count")
axs[1].set_title("Spectral Centroid (MIDI) Distribution")

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