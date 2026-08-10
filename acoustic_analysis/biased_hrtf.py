"""Elevation-biased HRTF on the KEMAR grid (hybrid with KEMAR azimuth).

Elevation timbre follows an artificial spectral bump (low elev → low frequencies,
high elev → high frequencies). Azimuth-dependent structure from KEMAR is kept as the
ratio KEMAR(az, el) / KEMAR(0, el) applied in the frequency domain per ear.
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin2
import slab

DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(os.path.dirname(DIR), "plots")
OUT_SOFA = os.path.join(DIR, "hrtf_elevation_bias.sofa")
OUT_PLOT = os.path.join(PLOT_DIR, "hrtf_elevation_bias.png")

# Peak frequency at minimum elevation (Hz); scales by 2**OCTAVES_SPAN at max elevation.
F_CENTER_AT_MIN_ELEV = 250.0
OCTAVES_SPAN = 5.5
LOG10_FREQ_SIGMA = 0.95  # width of elevation-dependent bump in log10(Hz)
BIAS_STRENGTH = 0.4  # 0 = flat spectrum; 1 = full elevation-dependent shaping
MAX_BUMP_ATTEN_DB = 30.0  # limit how deep the bump can dip away from its peak (dB)
RATIO_REG_EPS = 1e-3  # stabilizes KEMAR(az,el) / KEMAR(0,el) in frequency domain


def elevation_to_fir(
    elevation: float,
    elev_min: float,
    elev_max: float,
    n_taps: int,
    samplerate: float,
) -> np.ndarray:
    """FIR impulse response (n_taps × 2) with spectral peak set by elevation."""
    if elev_max <= elev_min:
        t = 0.5
    else:
        t = (elevation - elev_min) / (elev_max - elev_min)
    t = float(np.clip(t, 0.0, 1.0))
    f_center = F_CENTER_AT_MIN_ELEV * (2.0 ** (t * OCTAVES_SPAN))

    n_freq = n_taps // 2 + 1
    freqs = np.linspace(0.0, samplerate / 2.0, n_freq)
    log_f = np.log10(np.maximum(freqs, 1.0))
    log_fc = np.log10(f_center)
    bump_db = -20.0 * ((log_f - log_fc) / LOG10_FREQ_SIGMA) ** 2
    bump_db = np.maximum(bump_db, -MAX_BUMP_ATTEN_DB)
    bump = 10.0 ** (bump_db / 20.0)
    bump /= np.max(bump) + 1e-12
    mag = (1.0 - BIAS_STRENGTH) + BIAS_STRENGTH * bump
    mag[0] = mag[1]
    mag[-1] = 0.0  # Type II FIR (even tap count) requires zero Nyquist gain
    mag /= np.max(mag) + 1e-12

    h = firwin2(n_taps, freqs, mag, fs=samplerate)
    return np.column_stack([h, h])


def midline_source_index(hrtf: slab.HRTF, elevation: float) -> int:
    """Nearest midline (azimuth 0° cone) source for *elevation*."""
    indices = hrtf.cone_sources(0)
    vp = hrtf.sources.vertical_polar
    return min(indices, key=lambda i: abs(float(vp[i, 1]) - elevation))


def hybrid_filter_ir(
    bias_ir: np.ndarray,
    kemar_ir: np.ndarray,
    midline_ir: np.ndarray,
    reg_eps: float = RATIO_REG_EPS,
) -> np.ndarray:
    """B(el) * KEMAR(az, el) / KEMAR(0, el) in frequency domain (per ear)."""
    bias_h = np.fft.rfft(bias_ir, axis=0)
    kemar_h = np.fft.rfft(kemar_ir, axis=0)
    mid_h = np.fft.rfft(midline_ir, axis=0)
    scale = np.max(np.abs(mid_h)) + 1e-12
    hybrid_h = bias_h * kemar_h / (mid_h + reg_eps * scale)
    n_taps = bias_ir.shape[0]
    return np.fft.irfft(hybrid_h, n=n_taps, axis=0)


def build_elevation_biased_hrtf(template=None) -> slab.HRTF:
    """Hybrid HRTF: biased elevation on midline, KEMAR azimuth ratios off midline."""
    template = template or slab.HRTF.kemar()
    vp = template.sources.vertical_polar.astype(np.float64)
    elev_min, elev_max = vp[:, 1].min(), vp[:, 1].max()
    n_taps = template.data[0].n_samples
    sr = template.samplerate

    bias_by_elevation: dict[float, np.ndarray] = {}
    midline_by_elevation: dict[float, np.ndarray] = {}
    data = np.zeros((len(vp), n_taps, 2), dtype=np.float64)

    for i in range(len(vp)):
        el_key = round(float(vp[i, 1]), 1)
        if el_key not in bias_by_elevation:
            bias_by_elevation[el_key] = elevation_to_fir(
                el_key, elev_min, elev_max, n_taps, sr
            )
            mid_idx = midline_source_index(template, el_key)
            midline_by_elevation[el_key] = np.asarray(template.data[mid_idx].data)

        kemar_ir = np.asarray(template.data[i].data)
        data[i] = hybrid_filter_ir(
            bias_by_elevation[el_key],
            kemar_ir,
            midline_by_elevation[el_key],
        )

    return slab.HRTF(
        data,
        datatype="FIR",
        samplerate=sr,
        sources=vp,
        listener=copy.deepcopy(template.listener),
    )


def main():
    template = slab.HRTF.kemar()
    hrtf = build_elevation_biased_hrtf(template)
    hrtf.write_sofa(OUT_SOFA)
    vp = hrtf.sources.vertical_polar
    print(
        f"Wrote {OUT_SOFA} ({hrtf.n_sources} sources, {hrtf.datatype}, "
        f"elevation {vp[:, 1].min():.0f}° to {vp[:, 1].max():.0f}°)"
    )

    os.makedirs(PLOT_DIR, exist_ok=True)
    sourceidx = template.cone_sources(0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].set_title("Midline cone: image (left ear)")
    ax[1].set_title("Example magnitudes by elevation (midline)")

    hrtf.plot_tf(sourceidx, ear="left", axis=ax[0], show=False, kind="image")

    midline = template.cone_sources(0)
    vp_mid = template.sources.vertical_polar[midline]
    example_elevs = [-40, 0, 40, 80]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(example_elevs)))
    for elev, color in zip(example_elevs, colors):
        matches = [
            midline[j]
            for j in range(len(midline))
            if np.isclose(vp_mid[j, 1], elev, atol=0.05)
        ]
        if not matches:
            continue
        filt = hrtf.data[matches[0]]
        freqs, mag_db = filt.tf(show=False)
        ax[1].semilogx(freqs, mag_db[:, 0], color=color, label=f"{elev}°")
    ax[1].set_xlim(100, 18000)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    ax[1].legend(title="Elevation", fontsize=8)
    ax[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    plt.close()
    print(f"Wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()
