import csv
import os
import random

import numpy as np
import soundfile as sf

# ====== DIRS AND PARAMS
# Run from project root: python stimuli/select_test_set.py

DIR = os.getcwd()
STIM_ROOT = os.path.join(DIR, "stimuli")

ESC_DIR = os.path.join(STIM_ROOT, "ESC-50-master")
ESC_AUDIO_DIR = os.path.join(ESC_DIR, "audio")
ESC_META = os.path.join(ESC_DIR, "meta", "esc50.csv")
TINYSOL_DIR = os.path.join(STIM_ROOT, "TinySOL2020")
REFERENCE_DIR = os.path.join(STIM_ROOT, "naturalsounds165_tones")
NATURAL_REF_DIR = os.path.join(STIM_ROOT, "naturalsounds165")
OUT_DIR = os.path.join(STIM_ROOT, "test_stimuli")

# None -> same total number of sounds as naturalsounds165_tones
N_TOTAL = None
RANDOM_SEED = 42

# Match the format of naturalsounds165_tones
SAMPLE_RATE = 44100
DURATION = 2.0  # s
RAMP = 0.01  # s, raised-cosine on/off ramps
TARGET_RMS = 0.01
MIN_ACTIVE_RMS = 1e-4  # skip near-silent excerpts


def list_wavs(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".wav"))


def reference_proportion():
    """Fraction of natural sounds in naturalsounds165_tones, and its size."""
    natural = set(list_wavs(NATURAL_REF_DIR))
    ref = list_wavs(REFERENCE_DIR)
    n_natural = sum(f in natural for f in ref)
    return n_natural / len(ref), len(ref)


def allocate(groups, n):
    """Spread n picks as evenly as possible across groups (dict name -> items)."""
    names = list(groups)
    random.shuffle(names)
    counts = {name: 0 for name in names}
    remaining = n
    while remaining > 0:
        open_names = [k for k in names if counts[k] < len(groups[k])]
        if not open_names:
            raise ValueError(f"Not enough files to draw {n} sounds")
        for name in open_names[:remaining]:
            counts[name] += 1
        remaining -= min(remaining, len(open_names))
    return counts


def load_esc():
    groups = {}
    with open(ESC_META, newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["category"], []).append(
                os.path.join(ESC_AUDIO_DIR, row["filename"])
            )
    return groups


def load_tinysol():
    groups = {}
    for family in sorted(os.listdir(TINYSOL_DIR)):
        family_dir = os.path.join(TINYSOL_DIR, family)
        if not os.path.isdir(family_dir):
            continue
        for instrument in sorted(os.listdir(family_dir)):
            inst_dir = os.path.join(family_dir, instrument)
            if not os.path.isdir(inst_dir):
                continue
            wavs = [
                os.path.join(root, f)
                for root, _, files in os.walk(inst_dir)
                for f in files
                if f.endswith(".wav")
            ]
            if wavs:
                groups[instrument] = sorted(wavs)
    return groups


def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def loudest_window(x, n):
    """Excerpt of length n with the highest energy (ESC-50 clips are zero-padded)."""
    if len(x) <= n:
        return x
    energy = np.concatenate([[0.0], np.cumsum(x ** 2)])
    window_energy = energy[n:] - energy[:-n]
    start = int(np.argmax(window_energy))
    return x[start:start + n]


def finalize(x, n):
    """Pad to n samples, apply ramps and RMS-normalise."""
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    x = x[:n].copy()
    n_ramp = int(RAMP * SAMPLE_RATE)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_ramp)))
    x[:n_ramp] *= ramp
    x[-n_ramp:] *= ramp[::-1]
    rms = np.sqrt(np.mean(x ** 2))
    if rms < MIN_ACTIVE_RMS:
        return None
    return x * (TARGET_RMS / rms)


def process(path, source):
    x, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    x = to_mono(x)
    n = int(DURATION * SAMPLE_RATE)
    # Tones keep their onset; natural sounds use their most active part
    x = loudest_window(x, n) if source == "esc50" else x[:n]
    return finalize(x, n)


def draw(groups, n, source):
    """Draw n processed sounds balanced across groups, replacing silent picks."""
    counts = allocate(groups, n)
    pools = {k: random.sample(v, len(v)) for k, v in groups.items()}
    selected = []
    for group, k in counts.items():
        picked = 0
        while picked < k:
            if not pools[group]:
                raise ValueError(f"Ran out of usable files in {source}/{group}")
            path = pools[group].pop()
            audio = process(path, source)
            if audio is None:
                continue
            selected.append((group, path, audio))
            picked += 1
    return selected


# ====== BUILD TEST SET

if os.path.isdir(OUT_DIR) and os.listdir(OUT_DIR):
    raise FileExistsError(
        f"{OUT_DIR} already exists and is not empty. "
        "Remove or rename it before running this script."
    )

random.seed(RANDOM_SEED)

natural_fraction, ref_total = reference_proportion()
n_total = N_TOTAL or ref_total
n_natural = round(n_total * natural_fraction)
n_tones = n_total - n_natural
print(
    f"Reference: {natural_fraction:.1%} natural sounds -> "
    f"{n_natural} ESC-50 + {n_tones} TinySOL = {n_total} sounds"
)

esc = draw(load_esc(), n_natural, "esc50")
tinysol = draw(load_tinysol(), n_tones, "tinysol")

os.makedirs(OUT_DIR, exist_ok=True)
rows = []
for idx, (source, group, path, audio) in enumerate(
    [("esc50", *s) for s in esc] + [("tinysol", *s) for s in tinysol], start=1
):
    original = os.path.splitext(os.path.basename(path))[0].replace("#", "s")
    fname = f"test{idx:03d}_{source}_{group}_{original}.wav"
    sf.write(os.path.join(OUT_DIR, fname), audio, SAMPLE_RATE)
    rows.append({
        "filename": fname,
        "source": source,
        "category": group,
        "original_file": os.path.relpath(path, STIM_ROOT),
    })

with open(os.path.join(OUT_DIR, "test_stimuli.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"ESC-50: {len(esc)} sounds across {len({g for g, _, _ in esc})} categories")
print(f"TinySOL: {len(tinysol)} tones across {len({g for g, _, _ in tinysol})} instruments")
print(f"\nWrote {len(rows)} sounds to {OUT_DIR}")
