"""
Read every (stimulus name, azimuth, elevation) triple out of a cochleagram
TFRecord dataset and save it as a CSV.

Only reads the 'train/name' and 'train/target' features — never touches the
'train/image' cochleagram data, so this is fast even though each record's
image is ~14.6MB decompressed.
"""

import sys

sys.path.append("/home/neurobio/Repositories/RegressiveBinauralLocalizationCNN")

import pandas as pd
import tensorflow as tf
from blcnn.util import CNNpos_to_loc  # target int -> (azim, elev)
from tqdm import tqdm

# EDIT THESE:
PATH = "/home/neurobio/Repositories/RegressiveBinauralLocalizationCNN/data/cochleagrams/naturalsounds165_slab_kemar_front/test_cochleagrams.tfrecord"
OUT_CSV = "analysis_dnn/Results/stimulus_locations_test.csv"

feature_description = {
    "train/name": tf.io.FixedLenFeature([], tf.string),
    "train/target": tf.io.FixedLenFeature([], tf.int64),
}


def parse_name_and_target(raw):
    ex = tf.io.parse_single_example(raw, feature_description)
    return ex["train/name"], ex["train/target"]


def main():
    dataset = tf.data.TFRecordDataset(PATH, compression_type="GZIP")
    dataset = dataset.map(parse_name_and_target, num_parallel_calls=tf.data.AUTOTUNE)

    # sanity check on a handful before doing the full pass
    print("Sanity check on first 5 records:")
    for name, target in dataset.take(5):
        azim, elev = CNNpos_to_loc(int(target.numpy()))
        print(f"  {name.numpy().decode('utf-8')} -> azim {azim}, elev {elev}")

    # full pass
    rows = []
    for name, target in tqdm(dataset, desc="Reading name/target pairs"):
        name_str = name.numpy().decode("utf-8")
        target_int = int(target.numpy())
        azim, elev = CNNpos_to_loc(target_int)
        rows.append((name_str, target_int, azim, elev))

    df = pd.DataFrame(rows, columns=["stimulus", "target_code", "azim", "elev"])
    print(f"\n{len(df)} records read")

    # sanity check on the location grid against the generation config
    print("unique azims:", sorted(df.azim.unique()))
    print("unique elevs:", sorted(df.elev.unique()))

    # stimulus -> every (azim, elev) it was presented at
    locations_per_stim = df.groupby("stimulus")[["azim", "elev"]].apply(
        lambda g: list(zip(g["azim"], g["elev"]))
    )

    # how many locations per stimulus
    counts = df.groupby("stimulus").size()
    print("\nLocations per stimulus:")
    print(counts.describe())

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    return df, locations_per_stim


if __name__ == "__main__":
    df, locations_per_stim = main()

    # example lookups:
    # locations_per_stim.loc["stim298_contemporary_rock_song.wav"]
    # df[(df.azim == 90) & (df.elev == 0)]["stimulus"].unique()