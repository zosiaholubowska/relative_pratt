import slab
from slab.filter import Filter
import os
import numpy as np

DIR = os.getcwd()
STIM_DIR = os.path.join(DIR, 'stimuli')
OUTPUT_DIR = os.path.join(DIR, 'stimuli', 'artificial_sounds')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# general parameters

DURATION = 1.1
SAMPLING_RATE = 44100

# pink noise

pink_noise = slab.Sound.pinknoise(duration=DURATION, samplerate=SAMPLING_RATE, n_channels=2)
pink_noise = pink_noise.ramp(duration=0.01)
pink_noise.write(os.path.join(OUTPUT_DIR, 'pink_noise.wav'))

# narrow band noise

FILTER_BANK_PARAMS = {"n_filters": 10, "low_cutoff": 100}
_ref_sound = pink_noise

center_erb, _, erb_spacing = Filter._center_freqs(
    low_cutoff=FILTER_BANK_PARAMS["low_cutoff"],
    high_cutoff=_ref_sound.samplerate / 2,
    bandwidth=FILTER_BANK_PARAMS.get("bandwidth", 1 / 3),
    pass_bands=False,
    n_filters=FILTER_BANK_PARAMS["n_filters"],
)

_band_low_hz = Filter._erb2freq(center_erb - erb_spacing)
_band_high_hz = Filter._erb2freq(center_erb + erb_spacing)
_center_freqs_hz = Filter._erb2freq(center_erb)
_n_bands = len(_center_freqs_hz)

for band in range(_n_bands):
    band_noise = slab.Sound.equally_masking_noise(
        duration=DURATION,
        low_cutoff=_band_low_hz[band],
        high_cutoff=_band_high_hz[band],
        samplerate=SAMPLING_RATE,
    )
    # band_noise.spectrum(show=True)
    band_noise = band_noise.ramp(duration=0.01)
    band_noise.write(os.path.join(OUTPUT_DIR, f'band_{band}_noise.wav'))

# fixed kHz band-pass noise (<0.8; 0.8–1.4; 1.4–2.5; 2.5–4.5; 4.5–8; >8 kHz)

KHZ_BANDS = [
    ("lt_0.8", 100, 800),
    ("0.8_1.4", 800, 1400),
    ("1.4_2.5", 1400, 2500),
    ("2.5_4.5", 2500, 4500),
    ("4.5_8", 4500, 8000),
    ("gt_8", 8000, SAMPLING_RATE // 2),
]

for label, low_hz, high_hz in KHZ_BANDS:
    band_noise = slab.Sound.equally_masking_noise(
        duration=DURATION,
        low_cutoff=low_hz,
        high_cutoff=high_hz,
        samplerate=SAMPLING_RATE,
    )
    band_noise = band_noise.ramp(duration=0.01)
    band_noise.write(os.path.join(OUTPUT_DIR, f'khz_band_{label}_noise.wav'))

# ===== Pratt test sounds: pure tones, bright/dark harmonics, ERB-filtered pink noise

PRATT_OUTPUT_DIR = os.path.join(STIM_DIR, 'pratt_sounds')
os.makedirs(PRATT_OUTPUT_DIR, exist_ok=True)

RAMP_DURATION = 0.01
MIDI_NOTES = np.arange(55, 109)

# Harmonic levels in dB (relative to fundamental), cf. harmoniccomplex in utils.py.
HARMONIC_AMPLITUDES = {
    'bright': [0, -8, -11, -12, -11, -9, -7, -5],
    'dark': [0, -4, -10, -18, -30, -44, -58, -72],
}


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def bandpass_sound(sound, low_hz, high_hz):
    """Band-limit a sound with ideal FFT masking."""
    x = np.asarray(sound.data, dtype=float)
    sr = int(sound.samplerate)
    n_samples = x.shape[0]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    if x.ndim == 1:
        filtered = np.fft.irfft(np.fft.rfft(x) * mask, n=n_samples)
        return slab.Sound(filtered, samplerate=sr)

    filtered = np.empty_like(x)
    for ch in range(x.shape[1]):
        filtered[:, ch] = np.fft.irfft(np.fft.rfft(x[:, ch]) * mask, n=n_samples)
    return slab.Sound(filtered, samplerate=sr)


for midi in MIDI_NOTES:
    f0 = midi_to_hz(midi)

    pure = slab.Sound.tone(frequency=f0, duration=DURATION, samplerate=SAMPLING_RATE)
    pure = pure.ramp(duration=RAMP_DURATION)
    pure.write(os.path.join(PRATT_OUTPUT_DIR, f'stim_{midi}_pure.wav'))

    bright = slab.Sound.harmoniccomplex(
        f0=f0,
        duration=DURATION,
        amplitude=HARMONIC_AMPLITUDES['bright'],
        samplerate=SAMPLING_RATE,
    )
    bright = bright.ramp(duration=RAMP_DURATION)
    bright.write(os.path.join(PRATT_OUTPUT_DIR, f'stim_{midi}_bright_harmonic.wav'))

    dark = slab.Sound.harmoniccomplex(
        f0=f0,
        duration=DURATION,
        amplitude=HARMONIC_AMPLITUDES['dark'],
        samplerate=SAMPLING_RATE,
    )
    dark = dark.ramp(duration=RAMP_DURATION)
    dark.write(os.path.join(PRATT_OUTPUT_DIR, f'stim_{midi}_dark_harmonic.wav'))

_pink_ref = slab.Sound.pinknoise(duration=DURATION, samplerate=SAMPLING_RATE, n_channels=1)
for band in range(_n_bands):
    erb_pink = bandpass_sound(_pink_ref, _band_low_hz[band], _band_high_hz[band])
    erb_pink = erb_pink.ramp(duration=RAMP_DURATION)
    erb_pink.write(os.path.join(PRATT_OUTPUT_DIR, f'erb_pink_band_{band}.wav'))

print(f"Wrote Pratt sounds to {PRATT_OUTPUT_DIR}")
