import copy

import matplotlib.pyplot as plt
import numpy as np
import slab

import os

DIR = os.getcwd()

SYMMETRIC_ELEVATION_LIMIT = 40.0 # KEMAR has elevation range from -40° to +90°, so to make a flip, we need to crop to a symmetric range


def crop_hrtf_elevation(hrtf, min_elevation=-SYMMETRIC_ELEVATION_LIMIT, max_elevation=SYMMETRIC_ELEVATION_LIMIT):
    """Keep only sources within a symmetric elevation range."""
    vp = hrtf.sources.vertical_polar
    keep = np.where((vp[:, 1] >= min_elevation) & (vp[:, 1] <= max_elevation))[0]
    data = np.stack([hrtf.data[i].data.T for i in keep])
    return slab.HRTF(
        data,
        datatype=hrtf.datatype,
        samplerate=hrtf.samplerate,
        sources=vp[keep],
        listener=hrtf.listener,
    )


def flip_hrtf_elevation(hrtf):
    """Return an HRTF whose elevation dimension is inverted.

    A filter at elevation +θ uses the transfer function measured at elevation −θ
    (same azimuth). Every source must have an opposite-elevation counterpart;
    crop to a symmetric range first (e.g. −40° to +40°).
    """
    flipped = copy.deepcopy(hrtf)
    vp = hrtf.sources.vertical_polar

    for i in range(hrtf.n_sources):
        azimuth, elevation, _ = vp[i]
        mirror_idx = np.where(
            np.isclose(vp[:, 0], azimuth, atol=0.1)
            & np.isclose(vp[:, 1], -elevation, atol=0.1)
        )[0]
        if len(mirror_idx) == 0:
            raise ValueError(
                f"No mirror source for azimuth={azimuth:.2f}, elevation={elevation:.2f}. "
                f"Crop to a symmetric elevation range before flipping."
            )
        flipped.data[i] = copy.deepcopy(hrtf.data[mirror_idx[0]])

    return flipped


hrtf = crop_hrtf_elevation(slab.HRTF.kemar())
hrtf_flipped = flip_hrtf_elevation(hrtf)

sourceidx = hrtf.cone_sources(0)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].set_title("Original (−40° to +40°): waterfall")
ax[0, 1].set_title("Flipped: waterfall")
ax[1, 0].set_title("Original: image")
ax[1, 1].set_title("Flipped: image")

hrtf.plot_tf(sourceidx, ear="left", axis=ax[0, 0], show=False, kind="waterfall")
hrtf.plot_tf(sourceidx, ear="left", axis=ax[1, 0], show=False, kind="image")
hrtf_flipped.plot_tf(sourceidx, ear="left", axis=ax[0, 1], show=False, kind="waterfall")
hrtf_flipped.plot_tf(sourceidx, ear="left", axis=ax[1, 1], show=False, kind="image")

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(DIR, "plots","hrtf_flip.png"))


# Room simulation with flipped HRTF: pass hrtf= to room.hrir()
room = slab.Room(
    size=[4, 5, 3],
    listener=[2, 2.5, 1.4],
    source=[0, 25, 1.4],  # azimuth, elevation (°), distance (m) relative to listener
)

# binaural room impulse response (BRIR) 

brir_flipped = room.hrir(hrtf=hrtf_flipped, trim=5000)

sound = slab.Sound.pinknoise(duration=0.5, samplerate=hrtf.samplerate)
binaural_normal = room.hrir(hrtf=hrtf, trim=5000).apply(sound)
binaural_flipped = brir_flipped.apply(sound)

# in your cochleagram loop, replace room.hrir() with:
# brir = room.hrir(hrtf=hrtf_flipped, trim=5000)

