"""Flip KEMAR elevation axis (full grid): lowest direction gets highest filter, etc."""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import slab

DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(os.path.dirname(DIR), "plots")
OUT_SOFA = os.path.join(DIR, "hrtf_kemar_flipped.sofa")

AZIMUTH_ATOL = 0.1


def flip_hrtf_elevation(hrtf, azimuth_atol=AZIMUTH_ATOL):
    """Invert elevation per azimuth ring on the full HRTF grid.

    Source directions in the SOFA are unchanged. For each azimuth, filters
    are reversed along elevation: the filter assigned to the lowest elevation
    is the one measured at the highest elevation (same azimuth), and so on.
    Works on asymmetric ranges (e.g. KEMAR −40° to +90°) without cropping.
    """
    flipped = copy.deepcopy(hrtf)
    vp = hrtf.sources.vertical_polar
    azimuths = vp[:, 0]
    seen_rings: set[tuple[int, ...]] = set()

    for i in range(hrtf.n_sources):
        ring = np.where(np.isclose(azimuths, azimuths[i], atol=azimuth_atol))[0]
        key = tuple(sorted(ring.tolist()))
        if key in seen_rings:
            continue
        seen_rings.add(key)

        order = ring[np.argsort(vp[ring, 1])]
        n = len(order)
        for j, idx in enumerate(order):
            opposite = order[n - 1 - j]
            flipped.data[idx] = copy.deepcopy(hrtf.data[opposite])

    return flipped


def main():
    hrtf = slab.HRTF.kemar()
    hrtf_flipped = flip_hrtf_elevation(hrtf)
    hrtf_flipped.write_sofa(OUT_SOFA)
    vp = hrtf.sources.vertical_polar
    print(
        f"Wrote {OUT_SOFA} ({hrtf_flipped.n_sources} sources, "
        f"{hrtf_flipped.datatype}, elevation {vp[:, 1].min():.0f}° to {vp[:, 1].max():.0f}°)"
    )

    os.makedirs(PLOT_DIR, exist_ok=True)
    sourceidx = hrtf.cone_sources(0)
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].set_title("Original KEMAR: waterfall")
    ax[0, 1].set_title("Flipped: waterfall")
    ax[1, 0].set_title("Original: image")
    ax[1, 1].set_title("Flipped: image")

    hrtf.plot_tf(sourceidx, ear="left", axis=ax[0, 0], show=False, kind="waterfall")
    hrtf.plot_tf(sourceidx, ear="left", axis=ax[1, 0], show=False, kind="image")
    hrtf_flipped.plot_tf(sourceidx, ear="left", axis=ax[0, 1], show=False, kind="waterfall")
    hrtf_flipped.plot_tf(sourceidx, ear="left", axis=ax[1, 1], show=False, kind="image")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hrtf_flip.png"))
    plt.close()


if __name__ == "__main__":
    main()
