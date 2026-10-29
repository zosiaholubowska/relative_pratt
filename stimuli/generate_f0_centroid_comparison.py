import csv
import os

import librosa
import numpy as np
import soundfile as sf
from scipy.optimize import brentq

DIR = os.getcwd()
OUT_DIR = os.path.join(DIR, "stimuli", "f0_centroid_grid")

SAMPLE_RATE = 44100
DURATION = 1.5  # s
RAMP = 0.01  # s
TARGET_RMS = 0.01
RANDOM_SEED = 42

# librosa.load resamples to 22050 Hz by default, so keep energy below its Nyquist
F_MAX = 10000

F0_MIDI = np.linspace(57, 81, 5)  # 220-880 Hz
CENTROID_MIDI = np.linspace(93, 111, 10)  # 1760-4978 Hz
BANDWIDTH_OCT = 1.0  # SD of the log-frequency Gaussian envelope


def harmonics(f0, center):
    freqs = f0 * np.arange(1, int(F_MAX // f0) + 1)
    amps = np.exp(-0.5 * (np.log2(freqs / center) / BANDWIDTH_OCT) ** 2)
    return freqs, amps


def centroid(f0, center):
    freqs, amps = harmonics(f0, center)
    return np.sum(freqs * amps) / np.sum(amps)


def envelope_center(f0, target):
    """Envelope peak that gives exactly the target centroid at this f0."""
    return brentq(lambda c: centroid(f0, c) - target, 20, 40000)


def finalize(x):
    n_ramp = int(RAMP * SAMPLE_RATE)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_ramp)))
    x[:n_ramp] *= ramp
    x[-n_ramp:] *= ramp[::-1]
    return x * (TARGET_RMS / np.sqrt(np.mean(x ** 2)))


rng = np.random.default_rng(RANDOM_SEED)
phases = rng.uniform(0, 2 * np.pi, int(F_MAX // librosa.midi_to_hz(F0_MIDI.min())) + 1)
t = np.arange(int(DURATION * SAMPLE_RATE)) / SAMPLE_RATE

os.makedirs(OUT_DIR, exist_ok=True)
rows = []
idx = 0
for f0_midi in F0_MIDI:
    for c_midi in CENTROID_MIDI:
        idx += 1
        f0 = librosa.midi_to_hz(f0_midi)
        target = librosa.midi_to_hz(c_midi)
        freqs, amps = harmonics(f0, envelope_center(f0, target))
        x = np.zeros_like(t)
        for f, a, p in zip(freqs, amps, phases):
            x += a * np.sin(2 * np.pi * f * t + p)
        x = finalize(x)

        fname = f"grid{idx:03d}_f0-{f0_midi:.0f}_cent-{c_midi:.1f}.wav"
        path = os.path.join(OUT_DIR, fname)
        sf.write(path, x, SAMPLE_RATE)

        # Verify with the same measurement as acoustic_analysis/spectral_centroid_stimuli.py
        y, sr = librosa.load(path)
        measured = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rows.append({
            "filename": fname,
            "f0_hz": f0, "f0_midi": f0_midi,
            "centroid_target_hz": target, "centroid_target_midi": c_midi,
            "centroid_measured_hz": measured,
            "centroid_measured_midi": librosa.hz_to_midi(measured),
        })

with open(os.path.join(OUT_DIR, "f0_centroid_grid.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

m = np.array([[r["f0_midi"], r["centroid_measured_midi"]] for r in rows])
print(f"Wrote {len(rows)} sounds; r(f0, measured centroid) = {np.corrcoef(m.T)[0, 1]:.3f}")