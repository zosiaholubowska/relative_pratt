import os
import pandas as pd
import slab
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft
import librosa
import librosa.display
import librosa.feature
import seaborn as sns


plt.rcParams['svg.fonttype'] = 'none'

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'
TONE_DIR = f'{DIR}/stimuli/tones'
TRAINING_STIM_DIR = f'{DIR}/stimuli/natural-selection'


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

def rms_envelope(x, sr, win_ms=5, hop_ms=1):
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if win < 1:
        win = 1
    if hop < 1:
        hop = 1
    env = librosa.feature.rms(y=x, frame_length=win, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    return t, env

def attack_decay_time(x, sr, thresh_low=0.1, thresh_high=0.9):
    x = np.asarray(x)
    x = x - np.mean(x)
    env = np.abs(signal.hilbert(x))
    env = signal.savgol_filter(env, 101 if len(env) > 101 else max(5, len(env)//2*2+1), 3)
    peak = np.max(env)
    if peak <= 0:
        return np.nan, np.nan
    env = env / peak
    onset = np.where(env >= thresh_low)[0]
    peak_idx = np.argmax(env)
    if len(onset) == 0:
        return np.nan, np.nan
    attack_idx = onset[0]
    attack_time = (peak_idx - attack_idx) / sr if peak_idx > attack_idx else 0.0

    decay_region = env[peak_idx:]
    below = np.where(decay_region <= thresh_low)[0]
    if len(below) == 0:
        decay_time = np.nan
    else:
        decay_time = below[0] / sr
    return attack_time, decay_time

def spectral_centroid(x, sr):
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    return float(np.mean(cent))

def intensity_dbfs(x):
    rms = np.sqrt(np.mean(np.square(x)))
    return float(20 * np.log10(rms + 1e-12))

def vibrato_rate_depth(x, sr):
    y = np.asarray(x)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0 = pd.Series(f0).interpolate().bfill().ffill().to_numpy()
    if np.all(np.isnan(f0)) or np.nanstd(f0) == 0:
        return np.nan, np.nan
    cents = 1200 * np.log2(f0 / np.nanmean(f0))
    cents = cents[np.isfinite(cents)]
    if len(cents) < 8:
        return np.nan, np.nan
    fs_f0 = sr / 512
    f, Pxx = signal.welch(cents - np.mean(cents), fs=fs_f0, nperseg=min(256, len(cents)))
    mask = (f >= 4) & (f <= 12)
    if not np.any(mask):
        return np.nan, np.nan
    vib_rate = float(f[mask][np.argmax(Pxx[mask])])
    vib_depth = float(np.std(cents))
    return vib_rate, vib_depth

def harmonic_distribution(x, sr, n_harmonics=10):
    y = np.asarray(x)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0 = pd.Series(f0).interpolate().bfill().ffill().to_numpy()
    f0_mean = np.nanmean(f0)
    if not np.isfinite(f0_mean) or f0_mean <= 0:
        return [np.nan] * n_harmonics
    freqs = np.fft.rfftfreq(len(y), d=1/sr)
    mag = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    harmonic_amps = []
    for k in range(1, n_harmonics + 1):
        target = k * f0_mean
        idx = np.argmin(np.abs(freqs - target))
        harmonic_amps.append(float(mag[idx]))
    total = np.sum(harmonic_amps)
    if total > 0:
        harmonic_amps = [h / total for h in harmonic_amps]
    return harmonic_amps

# ================================================
# 2. COMPUTE ACOUSTIC FEATURES
# ================================================

# 1.1. Compute acoustic features for the tones used in the experiment
for tone in TONES[:4]:
    tone_path = os.path.join(TONE_DIR, tone)
    tone_sound = slab.Sound(tone_path)

    x = np.asarray(tone_sound.data).squeeze()
    x = np.squeeze(x)
    if x.ndim == 2:
        x = x.mean(axis=1)

    sr = int(tone_sound.samplerate)

    spec_cent = spectral_centroid(x, sr)
    vib_rate, vib_depth = vibrato_rate_depth(x, sr)
    attack_t, decay_t = attack_decay_time(x, sr)
    intensity = intensity_dbfs(x)
    harmonics = harmonic_distribution(x, sr, n_harmonics=10)

    row = {
        "file": tone,
        "spectral_centroid_hz": spec_cent,
        "vibrato_rate_hz": vib_rate,
        "vibrato_depth_cents": vib_depth,
        "attack_time_s": attack_t,
        "decay_time_s": decay_t,
        "intensity_dbfs": intensity,
    }

    for i, h in enumerate(harmonics, start=1):
        row[f"harmonic_{i}_rel_amp"] = h

    print(row)

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


PLOT_CONDITION = "harmoniccomplex"
PLOT_MIDI = 100
plot_tone_spectra(PLOT_CONDITION, PLOT_MIDI)
plt.savefig(f'{PLOT_DIR}/spectrogram_and_fft_{PLOT_CONDITION}_{PLOT_MIDI}.png', dpi=300)
plt.close()

