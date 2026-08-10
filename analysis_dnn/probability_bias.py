"""
For each naturalsounds165 stimulus, draw several elevations in [0, 60]°, with
selection probability biased toward lower elevation for low spectral centroid and
higher elevation for high centroid (inverse of the Parise-style spectrum–elevation mapping).
"""

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import slab

plt.rcParams["svg.fonttype"] = "none"

DIR = os.getcwd()
RESULTS_DIR = f"{DIR}/Results"
PLOT_DIR = f"{DIR}/plots"
TRAINING_STIM_DIR = f"{DIR}/stimuli/naturalsounds165"
CENTROID_CACHE = f"{RESULTS_DIR}/naturalsounds165_spectral_centroids.csv"
MANIFEST_PATH = f"{RESULTS_DIR}/naturalsounds165_probability_bias_manifest.csv"

ELEV_MIN = 0.0
ELEV_MAX = 60.0
N_LOCATIONS_PER_SOUND = 10
RNG_SEED = 42
# Gaussian width on centroid axis in the old sound-selection formulation; converted to ° on elevation
SIGMA_HZ = 1600.0

os.makedirs(RESULTS_DIR, exist_ok=True)


def spectral_centroid(x, sr):
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    return float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))


def load_mono(path):
    sound = slab.Sound(path)
    x = np.asarray(sound.data).squeeze()
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, int(sound.samplerate)


def build_centroid_table(stim_dir):
    stim_files = sorted(f for f in os.listdir(stim_dir) if f.startswith("stim") and f.endswith(".wav"))
    rows = []
    for fname in stim_files:
        path = os.path.join(stim_dir, fname)
        x, sr = load_mono(path)
        rows.append({"stimulus": fname, "spectral_centroid_hz": spectral_centroid(x, sr)})
    return pd.DataFrame(rows)


def elevation_to_target_centroid(elevation, elev_min, elev_max, cent_min, cent_max):
    elev = np.clip(elevation, elev_min, elev_max)
    t = (elev - elev_min) / (elev_max - elev_min)
    return cent_min + t * (cent_max - cent_min)


def centroid_to_target_elevation(centroid_hz, cent_min, cent_max, elev_min, elev_max):
    centroid_hz = np.clip(centroid_hz, cent_min, cent_max)
    t = (centroid_hz - cent_min) / (cent_max - cent_min)
    return elev_min + t * (elev_max - elev_min)


def hz_sigma_to_elev_sigma(sigma_hz, cent_min, cent_max, elev_min, elev_max):
    span_cent = cent_max - cent_min
    span_elev = elev_max - elev_min
    if span_cent <= 0:
        return float(sigma_hz)
    return float(sigma_hz * span_elev / span_cent)


def selection_probabilities(values, target, sigma):
    z = (np.asarray(values, dtype=float) - target) / sigma
    weights = np.exp(-0.5 * z ** 2)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def pick_elevations_for_sound(
    centroid_hz,
    n_locations,
    rng,
    cent_min,
    cent_max,
    elev_min=ELEV_MIN,
    elev_max=ELEV_MAX,
    sigma_elev=None,
    elevation_grid=None,
):
    if elevation_grid is None:
        elevation_grid = np.linspace(elev_min, elev_max, int(elev_max - elev_min) + 1)
    target_elev = centroid_to_target_elevation(
        centroid_hz, cent_min, cent_max, elev_min, elev_max
    )
    if sigma_elev is None:
        sigma_elev = hz_sigma_to_elev_sigma(SIGMA_HZ, cent_min, cent_max, elev_min, elev_max)
    probs = selection_probabilities(elevation_grid, target_elev, sigma_elev)
    chosen = rng.choice(elevation_grid, size=n_locations, p=probs)
    return chosen, float(target_elev)


if os.path.isfile(CENTROID_CACHE):
    centroid_df = pd.read_csv(CENTROID_CACHE)
else:
    centroid_df = build_centroid_table(TRAINING_STIM_DIR)
    centroid_df.to_csv(CENTROID_CACHE, index=False)

cent_min = float(centroid_df["spectral_centroid_hz"].min())
cent_max = float(centroid_df["spectral_centroid_hz"].max())
sigma_elev = hz_sigma_to_elev_sigma(SIGMA_HZ, cent_min, cent_max, ELEV_MIN, ELEV_MAX)

rng = np.random.default_rng(RNG_SEED)
manifest_rows = []

for _, sound_row in centroid_df.iterrows():
    stimulus = sound_row["stimulus"]
    centroid_hz = float(sound_row["spectral_centroid_hz"])
    elevations, target_elev = pick_elevations_for_sound(
        centroid_hz,
        N_LOCATIONS_PER_SOUND,
        rng,
        cent_min,
        cent_max,
        sigma_elev=sigma_elev,
    )
    for draw_idx, elevation in enumerate(elevations, start=1):
        manifest_rows.append(
            {
                "stimulus": stimulus,
                "spectral_centroid_hz": centroid_hz,
                "elevation": float(elevation),
                "draw_index": draw_idx,
            }
        )
    elev_str = ", ".join(f"{e:.0f}" for e in elevations)
    print(
        f"{stimulus}: centroid {centroid_hz:.0f} Hz, target elev {target_elev:.1f}° → "
        f"[{elev_str}]"
    )

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(MANIFEST_PATH, index=False)
print(
    f"\nWrote manifest with {len(manifest_df)} rows "
    f"({len(centroid_df)} sounds × {N_LOCATIONS_PER_SOUND} locations)"
)
print(f"Manifest: {MANIFEST_PATH}")

# --- diagnostic: selected elevation vs spectral centroid ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(
    manifest_df["spectral_centroid_hz"],
    manifest_df["elevation"],
    s=24,
    c=manifest_df["elevation"],
    cmap="inferno",
    edgecolors="k",
    linewidths=0.4,
    alpha=0.65,
    zorder=3,
)
c_line = np.linspace(cent_min, cent_max, 100)
target_line = centroid_to_target_elevation(c_line, cent_min, cent_max, ELEV_MIN, ELEV_MAX)
ax.plot(c_line, target_line, "k--", linewidth=1.2, label="Target elevation vs centroid")
ax.set_xlim(cent_min - 100, cent_max + 100)
ax.set_ylim(ELEV_MIN - 2, ELEV_MAX + 2)
ax.set_xlabel("Spectral centroid (Hz)")
ax.set_ylabel("Selected elevation (°)")
ax.set_title(
    f"Probability-biased elevation selection ({N_LOCATIONS_PER_SOUND} draws per sound)"
)
ax.legend(frameon=False)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/probability_bias_selection.svg", dpi=300)
fig.savefig(f"{PLOT_DIR}/probability_bias_selection.png", dpi=300)
plt.close(fig)
