"""
Compare frequency-channel energy distributions between two cochleagram datasets:
    naturalsounds165_slab_kemar
    naturalsounds165_slab_kemar_2

Usage:
    1. Set N_FREQ, N_TIME, N_EARS below to match your cochleagram shape
       (check the _summary_*.txt file in each data directory first --
       if the two datasets have DIFFERENT shapes, set them per-dataset
       in the DATASETS dict instead of using one global shape).
    2. Set N_SAMPLES to how many recordings to pull from each dataset
       (100-500 is usually plenty for a stable per-channel average).
    3. Run: python compare_cochleagram_frequencies.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only, no GPU needed for this

# ---- CONFIG ------------------------------------------------------------

BASE = '/home/neurobio/Repositories/RegressiveBinauralLocalizationCNN/data/cochleagrams'

DATASETS = {
    'slab_kemar':   {'dir': os.path.join(BASE, 'naturalsounds165_slab_kemar_1'),
                      'n_freq': 39, 'n_time': 8000, 'n_ears': 2},
    'slab_kemar_2': {'dir': os.path.join(BASE, 'naturalsounds165_slab_kemar_2'),
                      'n_freq': 39, 'n_time': 8000, 'n_ears': 2},
}

SPLIT = 'test'            # which file to open: 'train' or 'test'
FEATURE_KEY_PREFIX = 'train'  # prefix used INSIDE the tf.train.Example features
                               # (appears to always be 'train/...' regardless of
                               # which file/split you're reading -- confirmed by
                               # the 'test/image' KeyError when SPLIT='test')
COMPRESSION = 'GZIP'      # set to None if files turn out not to be gzip-compressed
N_SAMPLES = 200           # how many records to average over per dataset
DTYPE = np.float32        # matches how the bytes were written

PLOTS_DIR = os.path.join(os.getcwd(), 'plots')      # figures saved here
RESULTS_DIR = os.path.join(os.getcwd(), 'Results')  # CSVs saved here
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------


def load_frequency_profiles(data_dir, split, key_prefix, n_freq, n_time, n_ears,
                             compression, n_samples, dtype):
    """
    Reads up to n_samples records from {split}_cochleagrams.tfrecord,
    decodes the '{key_prefix}/image' bytes field, reshapes to
    (n_freq, n_time, n_ears), and returns:
        profiles: array (n_samples_actual, n_freq) -- mean energy per freq
                  channel per sample (averaged over time and ears)

    Note: `split` picks which FILE to open (train_cochleagrams.tfrecord vs
    test_cochleagrams.tfrecord). `key_prefix` is the prefix used INSIDE the
    tf.train.Example features (e.g. 'train/image') -- these are independent
    since some pipelines write every file's records with the same internal
    key prefix regardless of which file/split they end up in.
    """
    path = os.path.join(data_dir, f'{split}_cochleagrams.tfrecord')
    ds = tf.data.TFRecordDataset(path, compression_type=compression)

    expected_bytes = n_freq * n_time * n_ears * np.dtype(dtype).itemsize
    profiles = []
    image_key = f'{key_prefix}/image'

    for i, raw in enumerate(ds.take(n_samples)):
        ex = tf.train.Example.FromString(raw.numpy())
        feat = ex.features.feature

        if image_key not in feat or len(feat[image_key].bytes_list.value) == 0:
            available = list(feat.keys())
            raise KeyError(
                f"Sample {i} in {path}: key '{image_key}' not found or empty. "
                f"Available keys in this record: {available}. "
                f"Set FEATURE_KEY_PREFIX to match the prefix shown above."
            )
        img_bytes = feat[image_key].bytes_list.value[0]

        if len(img_bytes) != expected_bytes:
            raise ValueError(
                f"Sample {i} in {path}: got {len(img_bytes)} bytes, "
                f"expected {expected_bytes} for shape "
                f"({n_freq}, {n_time}, {n_ears}). Check your n_freq/n_time/n_ears "
                f"settings against the _summary_*.txt file for this dataset."
            )

        arr = np.frombuffer(img_bytes, dtype=dtype).reshape(n_freq, n_time, n_ears)
        # mean energy per frequency channel, averaged over time and both ears
        profiles.append(arr.mean(axis=(1, 2)))

    return np.array(profiles)  # shape: (n_samples_actual, n_freq)


def main():
    results = {}
    for name, cfg in DATASETS.items():
        print(f"Loading {name} from {cfg['dir']} ...")
        profiles = load_frequency_profiles(
            cfg['dir'], SPLIT, FEATURE_KEY_PREFIX, cfg['n_freq'], cfg['n_time'], cfg['n_ears'],
            COMPRESSION, N_SAMPLES, DTYPE
        )
        results[name] = profiles
        print(f"  -> loaded {profiles.shape[0]} samples, "
              f"{profiles.shape[1]} freq channels")

    # ---- Summary stats per frequency channel ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) mean +/- std per channel, overlaid
    for name, profiles in results.items():
        mean_per_ch = profiles.mean(axis=0)
        std_per_ch = profiles.std(axis=0)
        ch_idx = np.arange(len(mean_per_ch))
        axes[0].plot(ch_idx, mean_per_ch, label=name)
        axes[0].fill_between(ch_idx, mean_per_ch - std_per_ch,
                              mean_per_ch + std_per_ch, alpha=0.2)
    axes[0].set_xlabel('Frequency channel index')
    axes[0].set_ylabel('Mean energy (± 1 SD across samples)')
    axes[0].set_title('Per-channel energy: mean ± SD')
    axes[0].legend()

    # (b) distribution spread per channel as boxplots, low/mid/high bands
    n_freq = next(iter(results.values())).shape[1]
    bands = {
        'low':  slice(0, n_freq // 3),
        'mid':  slice(n_freq // 3, 2 * n_freq // 3),
        'high': slice(2 * n_freq // 3, n_freq),
    }
    band_names = list(bands.keys())
    positions = np.arange(len(band_names))
    width = 0.35

    for offset, (name, profiles) in zip([-width / 2, width / 2], results.items()):
        band_means = [profiles[:, s].mean(axis=1) for s in bands.values()]
        bp = axes[1].boxplot(band_means, positions=positions + offset,
                              widths=width, patch_artist=True, labels=band_names)
        for patch in bp['boxes']:
            patch.set_alpha(0.5)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(band_names)
    axes[1].set_ylabel('Mean energy per sample')
    axes[1].set_title('Low/mid/high frequency band energy by dataset')

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'cochleagram_frequency_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison figure to {out_path}")
    plt.show()

    # ---- Detailed numeric report ----
    names = list(results.keys())
    assert len(names) == 2, "This report section assumes exactly two datasets."
    name_a, name_b = names
    prof_a, prof_b = results[name_a], results[name_b]

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY (across all channels & samples)")
    print("=" * 70)
    for name, profiles in results.items():
        print(f"  {name:15s}: mean={profiles.mean():.4f}  std={profiles.std():.4f}  "
              f"n_samples={profiles.shape[0]}")

    overall_diff_pct = 100 * (prof_b.mean() - prof_a.mean()) / prof_a.mean()
    print(f"\n  Overall mean energy difference ({name_b} vs {name_a}): "
          f"{overall_diff_pct:+.1f}%")

    # ---- Per-channel comparison: mean, % difference, effect size, t-test ----
    print("\n" + "=" * 70)
    print("PER-CHANNEL COMPARISON")
    print("=" * 70)
    print(f"{'ch':>3} {'mean_'+name_a:>14} {'mean_'+name_b:>14} "
          f"{'%diff':>8} {'cohens_d':>9} {'p_value':>10}")

    mean_a = prof_a.mean(axis=0)
    mean_b = prof_b.mean(axis=0)
    n_freq = prof_a.shape[1]

    pooled_std = np.sqrt((prof_a.var(axis=0) + prof_b.var(axis=0)) / 2)
    cohens_d = np.divide(mean_b - mean_a, pooled_std,
                          out=np.zeros(n_freq), where=pooled_std != 0)
    pct_diff = 100 * np.divide(mean_b - mean_a, mean_a,
                                out=np.zeros(n_freq), where=mean_a != 0)

    p_values = np.array([
        stats.ttest_ind(prof_a[:, ch], prof_b[:, ch], equal_var=False).pvalue
        for ch in range(n_freq)
    ])

    for ch in range(n_freq):
        flag = " *" if p_values[ch] < 0.05 else ""
        print(f"{ch:>3} {mean_a[ch]:>14.4f} {mean_b[ch]:>14.4f} "
              f"{pct_diff[ch]:>+7.1f}% {cohens_d[ch]:>9.2f} {p_values[ch]:>10.4g}{flag}")

    n_sig = (p_values < 0.05).sum()
    print(f"\n  {n_sig}/{n_freq} channels differ significantly (p < 0.05, "
          f"Welch's t-test, uncorrected).")
    print("  NOTE: with many channels tested, consider a multiple-comparisons "
          "correction (e.g. Bonferroni: p < 0.05/{}={:.4g}) before drawing "
          "strong conclusions about individual channels.".format(n_freq, 0.05 / n_freq))

    # ---- Save per-channel results to CSV ----
    channel_csv_path = os.path.join(RESULTS_DIR, f'cochleagram_channel_comparison_{SPLIT}.csv')
    with open(channel_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', f'mean_{name_a}', f'mean_{name_b}',
                          'pct_diff', 'cohens_d', 'p_value', 'significant_p05'])
        for ch in range(n_freq):
            writer.writerow([ch, f'{mean_a[ch]:.6f}', f'{mean_b[ch]:.6f}',
                              f'{pct_diff[ch]:.2f}', f'{cohens_d[ch]:.3f}',
                              f'{p_values[ch]:.6g}', p_values[ch] < 0.05])
    print(f"\n  Saved per-channel results to {channel_csv_path}")

    # ---- Band-level comparison (coarser, easier to interpret) ----
    print("\n" + "=" * 70)
    print("FREQUENCY-BAND COMPARISON (low / mid / high thirds of channels)")
    print("=" * 70)
    band_rows = []
    for band_name, s in bands.items():
        band_a = prof_a[:, s].mean(axis=1)
        band_b = prof_b[:, s].mean(axis=1)
        t, p = stats.ttest_ind(band_a, band_b, equal_var=False)
        d = (band_b.mean() - band_a.mean()) / np.sqrt(
            (band_a.var() + band_b.var()) / 2)
        pct = 100 * (band_b.mean() - band_a.mean()) / band_a.mean()
        print(f"  {band_name:5s} (ch {s.start:>2}-{s.stop - 1:<2}): "
              f"{name_a}={band_a.mean():.4f}  {name_b}={band_b.mean():.4f}  "
              f"diff={pct:+.1f}%  cohens_d={d:.2f}  p={p:.4g}")
        band_rows.append([band_name, f'{s.start}-{s.stop - 1}',
                           f'{band_a.mean():.6f}', f'{band_b.mean():.6f}',
                           f'{pct:.2f}', f'{d:.3f}', f'{p:.6g}', p < 0.05])

    band_csv_path = os.path.join(RESULTS_DIR, 'cochleagram_band_comparison.csv')
    with open(band_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['band', 'channel_range', f'mean_{name_a}', f'mean_{name_b}',
                          'pct_diff', 'cohens_d', 'p_value', 'significant_p05'])
        writer.writerows(band_rows)
    print(f"\n  Saved band-level results to {band_csv_path}")

    # ---- Which channel differs most? ----
    top_ch = np.argsort(-np.abs(cohens_d))[:5]
    print("\n  Top 5 channels with largest effect size (|Cohen's d|):")
    for ch in top_ch:
        print(f"    channel {ch}: cohens_d={cohens_d[ch]:.2f}, "
              f"%diff={pct_diff[ch]:+.1f}%, p={p_values[ch]:.4g}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("""  - %diff and Cohen's d tell you the SIZE of the difference per channel/band
    (d ~0.2 small, ~0.5 medium, ~0.8+ large, by convention).
  - p-value tells you whether that difference is unlikely to be chance,
    given your sample size -- it does NOT tell you the difference is large
    or practically meaningful.
  - A channel can be "significant" (low p) but trivial in size if n_samples
    is large, or "non-significant" but large in size if n_samples is small.
    Read %diff/Cohen's d and p together, not p alone.
  - If differences cluster in the low or high band rather than being spread
    evenly, that points to a specific part of the auditory filterbank (e.g.
    HRTF/KEMAR model differences often show up more in high-frequency
    channels, which carry interaural level cues, vs low-frequency channels,
    which carry interaural timing cues).""")


if __name__ == '__main__':
    main()