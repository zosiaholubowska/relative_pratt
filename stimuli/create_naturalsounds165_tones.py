import os
import random
import shutil

# ====== DIRS AND PARAMS
# Run from project root: python stimuli/create_naturalsounds165_tones.py

DIR = os.getcwd()
STIM_ROOT = os.path.join(DIR, "stimuli")

NATURAL_DIR = os.path.join(STIM_ROOT, "naturalsounds165")
OUT_DIR = os.path.join(STIM_ROOT, "naturalsounds165_tones")
TONE_DIR = os.path.join(STIM_ROOT, "tones")

TONE_CONDITIONS = ["flute", "harmoniccomplex", "viola", "viola_complex"]
TONE_FRACTION = 0.5
RANDOM_SEED = 42


def list_wavs(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".wav"))


def sample_fraction(files, fraction):
    n = int(len(files) * fraction)
    if n == 0 and len(files) > 0:
        n = 1
    return random.sample(files, n)


# ====== BUILD BLENDED STIMULUS SET

if os.path.isdir(OUT_DIR) and os.listdir(OUT_DIR):
    raise FileExistsError(
        f"{OUT_DIR} already exists and is not empty. "
        "Remove or rename it before running this script."
    )

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

natural_files = list_wavs(NATURAL_DIR)
for fname in natural_files:
    shutil.copy2(os.path.join(NATURAL_DIR, fname), os.path.join(OUT_DIR, fname))

print(f"Copied {len(natural_files)} files from naturalsounds165")

tone_total = 0
for condition in TONE_CONDITIONS:
    src_dir = os.path.join(TONE_DIR, condition)
    wavs = list_wavs(src_dir)
    chosen = sample_fraction(wavs, TONE_FRACTION)
    for fname in chosen:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(OUT_DIR, fname))
    tone_total += len(chosen)
    print(f"{condition}: {len(chosen)} / {len(wavs)} tones (random {TONE_FRACTION:.0%})")

print(f"\nWrote {len(natural_files) + tone_total} sounds to {OUT_DIR}")
