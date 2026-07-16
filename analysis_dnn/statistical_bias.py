import numpy as np
import pandas 
import slab
import os
import matplotlib.pyplot as plt



plt.rcParams['svg.fonttype'] = 'none'

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'
TRAINING_STIM_DIR = f'{DIR}/stimuli/naturalsounds165'
OUTPUT_DIR = f'{DIR}/stimuli/naturalsounds165_scene_eq' 

TRAINING_TONES = [f for f in os.listdir(TRAINING_STIM_DIR) if 'stim' in f]

ELEVATIONS = np.arange(0, 70, 10)

EQ_KWARGS = dict(
    elev_min=0.0,
    elev_max=70.0,
    f_low=400.0,        
    f_high=6300.0,      
    sigma_octaves=1.2,  
    peak_gain_db=12,   # pilot: 3, 6, 9
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# functions

def elevation_to_center_hz(elevation, elev_min, elev_max, f_low=400.0, f_high=6300.0):
    """Map elevation to Gaussian center on a log-frequency axis (Parise: low elev → low freq)."""
    elev = np.clip(elevation, elev_min, elev_max)
    t = (elev - elev_min) / (elev_max - elev_min)
    log_fc = np.log2(f_low) + t * (np.log2(f_high) - np.log2(f_low))
    return 2.0 ** log_fc

def gaussian_log_gain(freqs_hz, center_hz, sigma_octaves, peak_gain_db):
    """
    Gaussian emphasis in log2-frequency.
    peak_gain_db: boost at center relative to skirts (before RMS normalization).
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    center_hz = max(center_hz, 1.0)
    log_f = np.log2(np.maximum(freqs_hz, 1.0))
    log_fc = np.log2(center_hz)
    gain_db = peak_gain_db * np.exp(-0.5 * ((log_f - log_fc) / sigma_octaves) ** 2)
    return 10.0 ** (gain_db / 20.0)

def apply_scene_eq(x, sr, elevation, elev_min=0.0, elev_max=70.0,
                   f_low=400.0, f_high=6300.0, sigma_octaves=1.2,
                   peak_gain_db=6.0):
    """
    Shape spectrum with elevation-dependent Gaussian emphasis; preserve RMS power.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    rms_orig = np.sqrt(np.mean(x ** 2))
    if rms_orig < 1e-12:
        return x
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = np.fft.rfft(x)
    center_hz = elevation_to_center_hz(elevation, elev_min, elev_max, f_low, f_high)
    gain = gaussian_log_gain(freqs, center_hz, sigma_octaves, peak_gain_db)
    y = np.fft.irfft(spectrum * gain, n=n)
    rms_new = np.sqrt(np.mean(y ** 2))
    if rms_new > 1e-12:
        y *= rms_orig / rms_new
    return y
    
def shape_training_sound(path, elevation, **eq_kwargs):
    """Load mono slab.Sound, apply scene EQ, return new slab.Sound."""
    sound = slab.Sound(path)
    x = np.asarray(sound.data)
    if x.ndim == 2:
        x_mono = x.mean(axis=1)
    else:
        x_mono = x
    y = apply_scene_eq(x_mono, int(sound.samplerate), elevation, **eq_kwargs)
    return slab.Sound(y, samplerate=sound.samplerate)

# ------------------------------------------------------------
# main loop for generating the sounds
# ------------------------------------------------------------

for tone in TRAINING_TONES:
    for elevation in ELEVATIONS:
        stim_path = os.path.join(TRAINING_STIM_DIR, tone)
        shaped = shape_training_sound(stim_path, elevation, **EQ_KWARGS)

        output_path = os.path.join(OUTPUT_DIR, f'{tone[:-4]}_{elevation}dB.wav')
        shaped.write(output_path)

# ------------------------------------------------------------
# plot the gaussian filters
# ------------------------------------------------------------

freqs_hz = np.logspace(np.log10(20), np.log10(EQ_KWARGS["f_high"]), 500)
colors = plt.cm.inferno(np.linspace(0, 1, len(ELEVATIONS)))

fig, ax = plt.subplots(figsize=(8, 5))
for elev, color in zip(ELEVATIONS, colors):
    center_hz = elevation_to_center_hz(
        elev,
        EQ_KWARGS["elev_min"],
        EQ_KWARGS["elev_max"],
        EQ_KWARGS["f_low"],
        EQ_KWARGS["f_high"],
    )
    gain_lin = gaussian_log_gain(
        freqs_hz,
        center_hz,
        EQ_KWARGS["sigma_octaves"],
        EQ_KWARGS["peak_gain_db"],
    )
    gain_db = 20.0 * np.log10(gain_lin)
    ax.plot(
        freqs_hz,
        gain_db,
        color=color,
        linewidth=2,
        label=f"{elev:.0f}° (centre {center_hz:.0f} Hz)",
    )


ax.set_xlim(0, EQ_KWARGS["f_high"]+1000)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Filter gain (dB)")
ax.set_title("Elevation-dependent Gaussian scene-EQ filters")
ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
ax.legend(title="Elevation", frameon=False, fontsize=9)
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/scene_eq_gaussian_filters.svg", dpi=300)
fig.savefig(f"{PLOT_DIR}/scene_eq_gaussian_filters.png", dpi=300)
plt.close(fig)


# ------------------------------------------------------------
# plot an example of change in the sound between 0 and 60 degrees
# ------------------------------------------------------------

tone = TRAINING_TONES[130]
ELEVATIONS = [0, 30, 60]
SPECTRUM_TICKS_HZ = [200, 500, 1000, 2000, 4000, 8000]

fig, axs = plt.subplots(
    ncols=len(ELEVATIONS), 
    figsize=(25/2.54, 12/2.54)
)
if len(ELEVATIONS) == 1:
    axs = [axs]

for idx, elevation in enumerate(ELEVATIONS):
    STIM_NAME = f'{tone[:-4]}_{elevation}dB.wav'
    stim_path = os.path.join(OUTPUT_DIR, STIM_NAME)
    sound = slab.Sound(stim_path)
    sound.spectrum(show=False, axis=axs[idx])
    axs[idx].set_title(f'{elevation}°')
    axs[idx].set_xlabel('Frequency (Hz)')
    axs[idx].set_xticks(SPECTRUM_TICKS_HZ)
    axs[idx].set_xticklabels(SPECTRUM_TICKS_HZ)
    axs[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()