import os
import re
import pandas as pd
import slab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import librosa
import librosa.display
import librosa.feature
import seaborn as sns


plt.rcParams['svg.fonttype'] = 'none'

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'
TONE_DIR = f'{DIR}/stimuli/tones'
TRAINING_STIM_DIR = f'{DIR}/stimuli/naturalsounds165'


# ================================================
# 0. READ AUDIO NAMES
# ================================================

# Tones used in the experiment
subfolders = ['flute', 'harmoniccomplex', 'viola', 'viola_complex']
TONES = []
for folder in subfolders:
    subfolder_path = os.path.join(TONE_DIR, folder)
    if os.path.isdir(subfolder_path):
        files = [os.path.join(folder, f) for f in os.listdir(subfolder_path) if f.startswith('stim')]
        TONES.extend(files)


# Tones used for the training
TRAINING_TONES = [f for f in os.listdir(TRAINING_STIM_DIR) if 'stim' in f]

# ================================================
# 1. FUNCTIONS TO COMPUTE ACOUSTIC FEATURES
# ================================================

def spectral_centroid(x, sr):
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    return float(np.mean(cent))


def mono_sound(sound):
    x = np.asarray(sound.data).squeeze()
    if x.ndim == 2:
        x = x.mean(axis=1)
    return slab.Sound(x, samplerate=sound.samplerate)


def band_levels(sound, filter_params):
    mono = mono_sound(sound)
    fbank = slab.Filter.cos_filterbank(
        length=mono.n_samples,
        samplerate=mono.samplerate,
        **filter_params,
    )
    return fbank.apply(mono).level


def filterbank_center_freqs(length, samplerate, filter_params):
    fbank = slab.Filter.cos_filterbank(
        length=length,
        samplerate=samplerate,
        **filter_params,
    )
    freqs_hz, mag = fbank.tf(show=False)
    return np.asarray(freqs_hz)[np.argmax(np.asarray(mag), axis=0)]


def condition_color_by_midi(base_hex, midi, midi_min, midi_max, sat_range=(0.25, 1.0)):
    hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(base_hex))
    norm = (midi - midi_min) / (midi_max - midi_min)
    hsv[1] = sat_range[0] + norm * (sat_range[1] - sat_range[0])
    return mcolors.hsv_to_rgb(hsv)

# ================================================
# 2.1. PLOT DISTRIBUTION OF AVERAGE SPECTRAL CENTROID OF THE TRAINING TONES
# ================================================

spectral_centroids = []

for tfile in TRAINING_TONES:
    tpath = os.path.join(TRAINING_STIM_DIR, tfile)
    tsound = slab.Sound(tpath)
    x_ = np.asarray(tsound.data).squeeze()
    x_ = x_.mean(axis=1) if x_.ndim == 2 else x_
    sr_ = int(tsound.samplerate)
    spec_cent = spectral_centroid(x_, sr_)
    spectral_centroids.append(spec_cent)
    print(tfile)
    print(spec_cent)


plt.figure(figsize=(7, 4))
sns.histplot(spectral_centroids, bins=20, kde=True, color='skyblue')
plt.xlabel("Spectral centroid (Hz)")
plt.ylabel("Number of training tones")
plt.title("Distribution of average spectral centroid of training tones")
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/spectral_centroid_distribution.svg', dpi=300)
plt.savefig(f'{PLOT_DIR}/spectral_centroid_distribution.png', dpi=300)
plt.close()


# ================================================
# 2.1b. BAND-LEVEL DISTRIBUTION OF TRAINING TONES (COSINE FILTER BANK)
# ================================================

FILTER_BANK_PARAMS = {"n_filters": 10, "low_cutoff": 100}

# compute the reference fbank for plotting

_ref_sound = slab.Sound(os.path.join(TRAINING_STIM_DIR, TRAINING_TONES[0]))
_ref_fbank = slab.Filter.cos_filterbank(
    length=_ref_sound.n_samples,
    samplerate=_ref_sound.samplerate,
    **FILTER_BANK_PARAMS,
)
_freqs_hz, _mag = _ref_fbank.tf(show=False)
_center_freqs_hz = np.asarray(_freqs_hz)[np.argmax(np.asarray(_mag), axis=0)]
_n_bands = len(_center_freqs_hz)

# analyze the training tones

band_level_rows = []
for tfile in TRAINING_TONES:
    tpath = os.path.join(TRAINING_STIM_DIR, tfile)
    tsound = slab.Sound(tpath)
    fbank = slab.Filter.cos_filterbank(
        length=tsound.n_samples,
        samplerate=tsound.samplerate,
        **FILTER_BANK_PARAMS,
    )
    levels = fbank.apply(tsound).level
    for band_idx, level in enumerate(levels):
        band_level_rows.append(
            {
                "stimulus": tfile,
                "band": band_idx,
                "center_freq_hz": float(_center_freqs_hz[band_idx]),
                "level_db": float(level),
            }
        )

band_levels_df = pd.DataFrame(band_level_rows)
band_levels_df.to_csv(f"{RESULTS_DIR}/training_tones_band_levels.csv", index=False)

# plot the band-level distribution

band_labels = [
    f"{cf / 1000:.2f}" if cf < 1000 else f"{cf / 1000:.1f}"
    for cf in _center_freqs_hz
]

violin_offset = -0.27
strip_offset = 0.27
violin_width = 0.5
box_width = 0.12


label_fontsize = 20
tick_fontsize = 18

fig, ax = plt.subplots(figsize=(max(10, _n_bands * 0.55), 5))
band_positions = np.arange(_n_bands)

violin_data = [
    band_levels_df.loc[band_levels_df["band"] == band_idx, "level_db"].to_numpy()
    for band_idx in band_positions
]
violin_parts = ax.violinplot(
    violin_data,
    positions=band_positions + violin_offset,
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

sns.boxplot(
    data=band_levels_df,
    x="band",
    y="level_db",
    order=band_positions,
    width=box_width,
    color="white",
    linewidth=1,
    fliersize=0,
    boxprops={"edgecolor": "black", "facecolor": "white"},
    whiskerprops={"color": "black"},
    capprops={"color": "black"},
    medianprops={"color": "black", "linewidth": 1.5},
    ax=ax,
)
ax.set_xticks(band_positions)
ax.set_xticklabels(band_labels, rotation=45, ha="right", fontsize=tick_fontsize)
ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
ax.set_xlabel("Frequency band center (kHz)", fontsize=label_fontsize)
ax.set_ylabel("Band level (dB SPL)", fontsize=label_fontsize)
ax.set_xlim(-0.6, _n_bands - 0.4)
ax.tick_params(axis='y', labelsize=tick_fontsize)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/training_tones_band_levels.svg", dpi=300)
fig.savefig(f"{PLOT_DIR}/training_tones_band_levels.png", dpi=300)
plt.close(fig)


# ================================================
# 2.2. PLOT SPECTROGRAM AND FFT OF A TONE
# ================================================

CONDITION_SUFFIX = {
    "flute": "flute",
    "harmoniccomplex": "harmonic",
    "viola": "viola",
    "viola_complex": "viola_complex",
}
def plot_tone_spectra(condition, midi):
    suffix = CONDITION_SUFFIX[condition]
    path = os.path.join(TONE_DIR, condition, f"stim_{midi}_{suffix}.wav")
    sound = slab.Sound(path)
    x = np.asarray(sound.data).squeeze()
    if x.ndim == 2:
        x = x.mean(axis=1)
    sr = int(sound.samplerate)

    fig, (ax_spec, ax_fft) = plt.subplots(1, 2, figsize=(12, 4))

    spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    img = librosa.display.specshow(spec_db, sr=sr, x_axis="time", y_axis="log", ax=ax_spec)
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    ax_spec.set_title(f"Spectrogram — {condition}, MIDI {midi}")

    freqs = np.fft.rfftfreq(len(x), d=1 / sr)
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    ax_fft.plot(freqs, mag, color="0.35", linewidth=0.9)
    ax_fft.set_xlim(0, sr / 2)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_title("FFT magnitude")

    fig.tight_layout()
    return fig

# USAGE:
# PLOT_CONDITION = "harmoniccomplex"
# PLOT_MIDI = 56
# plot_tone_spectra(PLOT_CONDITION, PLOT_MIDI)
# plt.savefig(f'{PLOT_DIR}/stimuli/spectrogram_and_fft_{PLOT_CONDITION}_{PLOT_MIDI}.png', dpi=300)
# plt.close()

for condition in CONDITION_SUFFIX.keys():
    for midi in [56, 73, 90]:
        plot_tone_spectra(condition, midi)
        plt.savefig(f'{PLOT_DIR}/stimuli/{condition}_{midi}.png', dpi=300)
        plt.close()

# ================================================
# 2.2b. BAND-LEVEL ANALYSIS OF EXPERIMENTAL TONES
# ================================================

CONDITION_PALETTE = {
    "flute": "#e22c1f",                # red
    "harmoniccomplex": "#33c33c",      # green 
    "viola": "#2e33a6",                # blue
    "viola_complex": "#ffd600",        # yellow 
}
STIM_PATH_RE = re.compile(r"^(?P<condition>[^/]+)/stim_(?P<midi>\d+)_")

_ref_exp_sound = slab.Sound(os.path.join(TONE_DIR, TONES[0]))
_exp_center_freqs_hz = filterbank_center_freqs(
    _ref_exp_sound.n_samples,
    _ref_exp_sound.samplerate,
    FILTER_BANK_PARAMS,
)
_exp_n_bands = len(_exp_center_freqs_hz)
_exp_band_labels = [
    f"{cf / 1000:.2f}" if cf < 1000 else f"{cf / 1000:.1f}"
    for cf in _exp_center_freqs_hz
]
_exp_band_positions = np.arange(_exp_n_bands)

exp_band_rows = []
for tone_rel_path in TONES:
    match = STIM_PATH_RE.match(tone_rel_path)
    if not match:
        continue
    condition = match.group("condition")
    midi = int(match.group("midi"))
    tone_path = os.path.join(TONE_DIR, tone_rel_path)
    sound = slab.Sound(tone_path)
    levels = band_levels(sound, FILTER_BANK_PARAMS)
    for band_idx, level in enumerate(levels):
        exp_band_rows.append(
            {
                "condition": condition,
                "midi": midi,
                "band": band_idx,
                "center_freq_hz": float(_exp_center_freqs_hz[band_idx]),
                "level_db": float(level),
            }
        )

exp_band_levels_df = pd.DataFrame(exp_band_rows)
exp_band_levels_df.to_csv(f"{RESULTS_DIR}/experiment_tones_band_levels.csv", index=False)

midi_min = exp_band_levels_df["midi"].min()
midi_max = exp_band_levels_df["midi"].max()
label_fontsize = 20
tick_fontsize = 18

# heatmap: MIDI × frequency band, one panel per condition

heatmap_matrices = {}
for condition in CONDITION_SUFFIX:
    cond_df = exp_band_levels_df.loc[exp_band_levels_df["condition"] == condition]
    heatmap_matrices[condition] = cond_df.pivot(
        index="midi", columns="band", values="level_db"
    ).sort_index()

level_vmin = exp_band_levels_df["level_db"].min()
level_vmax = exp_band_levels_df["level_db"].max()

fig, axes = plt.subplots(2, 2, figsize=(max(10, _exp_n_bands * 0.7) * 1.6, 12))
axes = axes.flatten()
heatmap_img = None

for idx, (ax, condition) in enumerate(zip(axes, CONDITION_SUFFIX)):
    heatmap_img = ax.imshow(
        heatmap_matrices[condition],
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=level_vmin,
        vmax=level_vmax,
        extent=(-0.5, _exp_n_bands - 0.5, midi_min - 0.5, midi_max + 0.5),
    )
    ax.set_title(condition, fontsize=label_fontsize)
    ax.set_xticks(_exp_band_positions)
    ax.set_xticklabels(_exp_band_labels, rotation=45, ha="right", fontsize=tick_fontsize)
    # Only label x-axis for bottom row and y-axis for left column
    nrows, ncols = 2, 2
    row, col = divmod(idx, ncols)
    if row == nrows - 1:
        ax.set_xlabel("Frequency band center (kHz)", fontsize=label_fontsize)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("MIDI note", fontsize=label_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

cbar = fig.colorbar(heatmap_img, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label("Band level (dB SPL)", fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)
fig.savefig(f"{PLOT_DIR}/experiment_tones_band_levels_heatmap.svg", dpi=300, bbox_inches="tight")
fig.savefig(f"{PLOT_DIR}/experiment_tones_band_levels_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ================================================
# 3. KEMAR HRTF ANALYSIS (PARISE-STYLE)
# ================================================

KEMAR_PLOT_DIR = f"{PLOT_DIR}/kemar"
os.makedirs(KEMAR_PLOT_DIR, exist_ok=True)

# Parise-style frequency bands (Hz): lower inclusive, upper exclusive
PARISE_BANDS_HZ = [
    (0, 800, "<0.8"),
    (800, 1400, "0.8–1.4"),
    (1400, 2500, "1.4–2.5"),
    (2500, 4500, "2.5–4.5"),
    (4500, 8000, "4.5–8"),
    (8000, 16000, "8-16"),
    (16000, 32000, "16-32"),
]
EXPERIMENTAL_ELEVATIONS = [-25.0, -12.5, 0.0, 12.5, 25.0]
ELEVATION_GRID = np.arange(-40, 41, 1)


def _band_mask(freqs, f_low, f_high):
    if f_high is None:
        return freqs >= f_low
    return (freqs >= f_low) & (freqs < f_high)

def hrtf_transfer_db(hrtf, azimuth, elevation):
    """Return frequency axis and HRTF magnitude (dB) for left and right ears."""
    filt = hrtf.interpolate(azimuth=azimuth, elevation=elevation)
    freqs, mag_db = filt.tf(show=False)
    return freqs, mag_db


def mean_gain_in_band(mag_db, freqs, f_low, f_high):
    mask = _band_mask(freqs, f_low, f_high)
    if not np.any(mask):
        return np.nan
    return float(np.mean(mag_db[mask]))


def band_rms(x, sr, f_low, f_high):
    """RMS of the time-domain signal after ideal bandpass (FFT masking)."""
    x = np.asarray(x, dtype=float)
    freqs = np.fft.rfftfreq(len(x), d=1 / sr)
    spectrum = np.fft.rfft(x)
    mask = _band_mask(freqs, f_low, f_high)
    filtered = np.fft.irfft(spectrum * mask, n=len(x))
    return float(np.sqrt(np.mean(filtered ** 2)))


hrtf = slab.HRTF.kemar()
sound = slab.Sound.pinknoise(duration=1.0, samplerate=hrtf.samplerate)
nyquist = hrtf.samplerate / 2
parise_bands = [
    (f_low, nyquist if f_high is None else f_high, label)
    for f_low, f_high, label in PARISE_BANDS_HZ
]

# --- Build HRTF magnitude surface on the midsagittal plane ---
freqs, _ = hrtf_transfer_db(hrtf, azimuth=0, elevation=0)
mag_stack = np.empty((len(ELEVATION_GRID), len(freqs), 2))

for elev_idx, elev in enumerate(ELEVATION_GRID):
    _, mag_db = hrtf_transfer_db(hrtf, azimuth=0, elevation=elev)
    mag_stack[elev_idx] = mag_db

# Gain at each experimental location, per frequency band
gain_rows = []
for elev in EXPERIMENTAL_ELEVATIONS:
    _, mag_db = hrtf_transfer_db(hrtf, azimuth=0, elevation=elev)
    mag_mean = mag_db.mean(axis=1)
    row = {"elevation": elev}
    for f_low, f_high, label in parise_bands:
        row[f"gain_db_{label}"] = mean_gain_in_band(mag_mean, freqs, f_low, f_high)
    gain_rows.append(row)

gain_df = pd.DataFrame(gain_rows).set_index("elevation")
gain_df.to_csv(f"{RESULTS_DIR}/kemar_band_gain_by_elevation.csv")

# Plot 1 — HRTF band gain at each experimental elevation
band_labels = [label for _, _, label in parise_bands]
x_pos = np.arange(len(band_labels))

fig, ax = plt.subplots(figsize=(7, 4.5))
for elev in EXPERIMENTAL_ELEVATIONS:
    gains = gain_df.loc[elev, [f"gain_db_{label}" for label in band_labels]].to_numpy()
    ax.plot(x_pos, gains, "o-", linewidth=2, markersize=6, label=f"{elev:+.1f}°")

ax.set_xticks(x_pos)
ax.set_xticklabels(band_labels, rotation=25, ha="right")
ax.set_xlabel("Frequency band (kHz)")
ax.set_ylabel("Mean HRTF gain (dB)")
ax.set_title("KEMAR band gain by experimental elevation")
ax.legend(title="Elevation", frameon=False, loc="best")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{KEMAR_PLOT_DIR}/hrtf_band_gain_by_elevation.svg", dpi=300)
fig.savefig(f"{KEMAR_PLOT_DIR}/hrtf_band_gain_by_elevation.png", dpi=300)
plt.close(fig)

# Band-limited RMS of HRTF-filtered pink noise at each experimental elevation
rms_rows = []
for elev in EXPERIMENTAL_ELEVATIONS:
    filt = hrtf.interpolate(azimuth=0, elevation=elev)
    binaural = filt.apply(sound)
    sr = binaural.samplerate
    mono = binaural.data.mean(axis=1)
    row = {"elevation": elev}
    for f_low, f_high, label in parise_bands:
        row[f"rms_{label}"] = band_rms(mono, sr, f_low, f_high)
    rms_rows.append(row)

rms_df = pd.DataFrame(rms_rows).set_index("elevation")
rms_df.to_csv(f"{RESULTS_DIR}/kemar_pinknoise_band_rms_by_elevation.csv")

fig, ax = plt.subplots(figsize=(7, 4.5))
for elev in EXPERIMENTAL_ELEVATIONS:
    rms_vals = rms_df.loc[elev, [f"rms_{label}" for label in band_labels]].to_numpy()
    ax.plot(x_pos, rms_vals, "o-", linewidth=2, markersize=6, label=f"{elev:+.1f}°")

ax.set_xticks(x_pos)
ax.set_xticklabels(band_labels, rotation=25, ha="right")
ax.set_xlabel("Frequency band (kHz)")
ax.set_ylabel("Band-limited RMS")
ax.set_title("HRTF-filtered pink noise: band RMS by experimental elevation")
ax.legend(title="Elevation", frameon=False, loc="best")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{KEMAR_PLOT_DIR}/pinknoise_band_rms_by_elevation.svg", dpi=300)
fig.savefig(f"{KEMAR_PLOT_DIR}/pinknoise_band_rms_by_elevation.png", dpi=300)
plt.close(fig)

