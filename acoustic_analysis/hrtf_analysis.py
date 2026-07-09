"""
Visualize KEMAR HRTF filtering with image plots.

Run:
    python hrtf_analysis.py
"""
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import slab

plt.rcParams["svg.fonttype"] = "none"

DIR = os.getcwd()
PLOT_DIR = f"{DIR}/plots"
TONE_PATH = f"{DIR}/stimuli/tones/flute/stim_73_flute.wav"
ELEVATIONS = [-25.0, 0.0, 25.0]


def plot_hrtf_waterfall_and_image(cone=0, out_name="kemar_hrtf_waterfall_image"):
    """Reproduce the slab example: waterfall and image plot of HRTF transfer functions."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    hrtf = slab.HRTF.kemar()
    sourceidx = hrtf.cone_sources(cone)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].set_title("waterfall plot")
    ax[1].set_title("image plot")
    hrtf.plot_tf(sourceidx, ear="left", axis=ax[0], show=False, kind="waterfall")
    hrtf.plot_tf(sourceidx, ear="left", axis=ax[1], show=False, kind="image")
    plt.tight_layout()

    png_path = f"{PLOT_DIR}/{out_name}.png"
    svg_path = f"{PLOT_DIR}/{out_name}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


def plot_kemar_filtered_image(
    tone_path=TONE_PATH,
    elevations=ELEVATIONS,
    out_name="kemar_filtered_image_flute_73",
):
    """
    Show:
    - top: KEMAR HRTF image plot (cone_sources) with elevation markers
    - bottom: dry and KEMAR-filtered spectrograms
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    hrtf = slab.HRTF.kemar()

    tone = slab.Sound(tone_path)
    if tone.data.ndim > 1:
        tone = slab.Sound(tone.data.mean(axis=1), samplerate=tone.samplerate)
    tone = tone.resample(hrtf.samplerate)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.35, wspace=0.25)

    # Top: full HRTF image along frontal cone (azimuth = 0)
    ax_hrtf = fig.add_subplot(gs[0, :])
    sourceidx = hrtf.cone_sources(0)
    hrtf.plot_tf(
        sourceidx,
        ear="left",
        axis=ax_hrtf,
        show=False,
        kind="image",
        xlim=(200, 8000),
    )
    for elev in elevations:
        ax_hrtf.axhline(elev, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
        ax_hrtf.text(8200, elev, f"{elev:+.0f}°", color="white", va="center", fontsize=9)
    ax_hrtf.set_title("KEMAR HRTF image plot (left ear, frontal azimuth = 0°)")
    ax_hrtf.set_xlabel("Frequency [Hz]")
    ax_hrtf.set_ylabel("Elevation [°]")

    # Bottom: dry + filtered spectrograms
    x = np.asarray(tone.data).squeeze()
    sr = int(tone.samplerate)
    spec_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    dry_spec = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    librosa.display.specshow(
        dry_spec,
        sr=sr,
        x_axis="time",
        y_axis="log",
        ax=spec_axes[0],
        cmap="magma",
    )
    spec_axes[0].set_title("Dry sound")

    for ax, elevation in zip(spec_axes[1:], elevations):
        filt = hrtf.interpolate(azimuth=0, elevation=elevation)
        filtered = filt.apply(tone)
        left = filtered.data[:, 0]
        filt_spec = librosa.amplitude_to_db(np.abs(librosa.stft(left)), ref=np.max)
        librosa.display.specshow(
            filt_spec,
            sr=sr,
            x_axis="time",
            y_axis="log",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(f"KEMAR filtered\nelev {elevation:+.0f}° (left ear)")

    fig.suptitle(
        "Flute MIDI 73: KEMAR HRTF image plot and filtered-sound spectrograms",
        y=1.01,
    )

    png_path = f"{PLOT_DIR}/{out_name}.png"
    svg_path = f"{PLOT_DIR}/{out_name}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    plot_hrtf_waterfall_and_image()
    plot_kemar_filtered_image()
