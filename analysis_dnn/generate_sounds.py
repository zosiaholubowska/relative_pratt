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