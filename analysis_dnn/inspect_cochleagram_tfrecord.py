"""
Inspect (and optionally compare) cochleagram TFRecord files.

Uses the `tfrecord` PyPI package (no full TensorFlow install required):
    pip install tfrecord

Examples:
    python inspect_cochleagram_tfrecord.py /path/to/train.tfrecord
    python inspect_cochleagram_tfrecord.py a.tfrecord --compare b.tfrecord --plot-key cochleagram
    python inspect_cochleagram_tfrecord.py data.tfrecord --index data.tfrecord.index --gzip
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from tfrecord import example_pb2, tfrecord_iterator
    from tfrecord.reader import extract_feature_dict
except ImportError as exc:
    raise SystemExit(
        "Install the reader: pip install tfrecord\n"
        "Or use TensorFlow: tf.data.TFRecordDataset(...)"
    ) from exc

plt.rcParams["svg.fonttype"] = "none"

COCHLEAGRAM_KEY_HINTS = (
    "cochleagram",
    "cochleagrams",
    "gram",
    "spectrogram",
    "spec",
    "audio",
    "waveform",
    "embedding",
)


def try_parse_example(record: memoryview) -> example_pb2.Example | None:
    ex = example_pb2.Example()
    try:
        ex.ParseFromString(record)
        if ex.features.feature:
            return ex
    except Exception:
        pass
    return None


def try_parse_sequence_example(record: memoryview):
    seq = example_pb2.SequenceExample()
    try:
        seq.ParseFromString(record)
        if seq.context.feature or seq.feature_lists.feature_list:
            return seq
    except Exception:
        pass
    return None


def summarize_feature(name: str, value) -> str:
    if isinstance(value, bytes):
        nbytes = len(value)
        for dtype in (np.float32, np.float64, np.int32, np.int64):
            if nbytes % dtype().nitemsize == 0:
                n = nbytes // dtype().nitemsize
                return f"bytes len={nbytes} (fits {n} × {dtype.__name__})"
        return f"bytes len={nbytes}"
    if isinstance(value, np.ndarray):
        return f"array shape={value.shape} dtype={value.dtype}"
    return repr(value)[:120]


def bytes_to_array(raw: bytes, dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=dtype)
    return np.array(arr, copy=True)


def infer_shape_from_features(features: dict) -> tuple[int, ...] | None:
    """Common TFRecord patterns: explicit shape_* keys or a single shape vector."""
    shape_keys = [k for k in features if "shape" in k.lower()]
    if "shape" in features:
        s = np.asarray(features["shape"]).astype(int).ravel()
        if s.size:
            return tuple(int(x) for x in s)
    dim_keys = sorted(k for k in features if k.lower().startswith("shape_"))
    if dim_keys:
        return tuple(int(np.asarray(features[k]).ravel()[0]) for k in dim_keys)
    for key in ("height", "width", "time", "n_frames", "n_freqs"):
        if key in features:
            pass
    h = features.get("height") or features.get("n_freqs")
    w = features.get("width") or features.get("time") or features.get("n_frames")
    if h is not None and w is not None:
        return (int(np.asarray(h).ravel()[0]), int(np.asarray(w).ravel()[0]))
    return None


def pick_cochleagram_key(features: dict, plot_key: str | None) -> str | None:
    if plot_key and plot_key in features:
        return plot_key
    for hint in COCHLEAGRAM_KEY_HINTS:
        for k in features:
            if hint in k.lower():
                return k
    for k, v in features.items():
        if isinstance(v, bytes) and len(v) > 1024:
            return k
        if isinstance(v, np.ndarray) and v.size > 1024 and v.dtype != np.int64:
            return k
    return None


def feature_to_2d(features: dict, key: str) -> np.ndarray | None:
    raw = features[key]
    if isinstance(raw, np.ndarray) and raw.dtype != np.bytes_ and raw.dtype.kind != "S":
        arr = np.asarray(raw, dtype=float)
    elif isinstance(raw, (bytes, bytearray)):
        arr = bytes_to_array(bytes(raw), dtype=np.float32)
    elif isinstance(raw, np.ndarray) and raw.dtype.kind in "SU":
        arr = bytes_to_array(bytes(raw.tobytes()), dtype=np.float32)
    else:
        return None

    shape = infer_shape_from_features(features)
    if shape and int(np.prod(shape)) == arr.size:
        arr = arr.reshape(shape)
    elif arr.size == 0:
        return None

    if arr.ndim == 1:
        n = int(np.sqrt(arr.size))
        if n * n == arr.size:
            arr = arr.reshape(n, n)
        else:
            arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1) if arr.shape[0] < arr.shape[-1] else arr.squeeze()
    return arr


def load_records(path: str, index_path: str | None, compression: str | None, limit: int):
    """Yield decoded feature dicts; try Example then SequenceExample."""
    typename_mapping = {"byte": "bytes_list", "float": "float_list", "int": "int64_list"}
    records = []
    for i, record in enumerate(tfrecord_iterator(path, index_path, compression_type=compression)):
        if i >= limit:
            break
        ex = try_parse_example(record)
        if ex is not None:
            records.append(extract_feature_dict(ex.features, None, typename_mapping))
            continue
        seq = try_parse_sequence_example(record)
        if seq is not None:
            ctx = extract_feature_dict(seq.context, None, typename_mapping)
            ctx["_record_kind"] = np.array(["sequence"])
            records.append(ctx)
            continue
        records.append({"_parse_error": np.array([1])})
    return records


def print_record_summary(path: str, features: dict, record_idx: int) -> None:
    print(f"\n=== {path}  record {record_idx} ===")
    if "_parse_error" in features:
        print("Could not parse as tf.train.Example or SequenceExample.")
        return
    if "_record_kind" in features:
        print("(SequenceExample — showing context features only; frames may be in feature_lists.)")
    for key in sorted(k for k in features if not k.startswith("_")):
        print(f"  {key}: {summarize_feature(key, features[key])}")


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(float) - b.astype(float)
    return {
        "shape_a": a.shape,
        "shape_b": b.shape,
        "max_abs_diff": float(np.max(np.abs(d))),
        "mean_abs_diff": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "pearson_r": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect cochleagram TFRecord files.")
    parser.add_argument("path", help="Path to .tfrecord (or .tfrecord.gz)")
    parser.add_argument(
        "--compare",
        help="Second TFRecord to compare (same record order / matching keys)",
    )
    parser.add_argument("--index", help="Optional .index file for random access")
    parser.add_argument(
        "--compare-index",
        help="Index for --compare file",
    )
    parser.add_argument("--gzip", action="store_true", help="Records are gzip-compressed")
    parser.add_argument(
        "--record",
        type=int,
        default=0,
        help="Record index to summarize and plot (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many records to scan for the initial summary (default: 3)",
    )
    parser.add_argument(
        "--plot-key",
        default=None,
        help="Feature name to visualize (auto-guess if omitted)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Save figure to this path (default: plots/tfrecord_inspect.png)",
    )
    args = parser.parse_args()

    compression = "gzip" if args.gzip or args.path.endswith(".gz") else None
    if not os.path.isfile(args.path):
        raise SystemExit(f"File not found: {args.path}")

    summaries = load_records(args.path, args.index, compression, args.limit)
    if not summaries:
        raise SystemExit("No records read — check path, --gzip, or --index.")

    for i, feat in enumerate(summaries):
        print_record_summary(args.path, feat, i)

    if args.record >= len(summaries):
        print(f"\nRecord {args.record} not in first --limit={args.limit}; loading one record…")
        summaries = load_records(args.path, args.index, compression, args.record + 1)
    features = summaries[min(args.record, len(summaries) - 1)]
    key = pick_cochleagram_key({k: v for k, v in features.items() if not k.startswith("_")}, args.plot_key)
    if key is None:
        print("\nNo plottable cochleagram-like feature found. Pass --plot-key explicitly.")
        return

    arr = feature_to_2d(features, key)
    if arr is None:
        raise SystemExit(f"Could not decode feature {key!r} as a numeric array.")

    compare_arr = None
    if args.compare:
        comp_compression = "gzip" if args.gzip or args.compare.endswith(".gz") else None
        comp_summaries = load_records(
            args.compare, args.compare_index, comp_compression, args.record + 1
        )
        if args.record >= len(comp_summaries):
            raise SystemExit(f"Compare file has fewer than {args.record + 1} records.")
        comp_features = comp_summaries[args.record]
        comp_key = pick_cochleagram_key(comp_features, args.plot_key or key)
        compare_arr = feature_to_2d(comp_features, comp_key)
        if compare_arr is None:
            raise SystemExit(f"Could not decode compare feature {comp_key!r}.")
        if compare_arr.shape != arr.shape:
            print(
                f"Warning: shape mismatch {arr.shape} vs {compare_arr.shape} — "
                "using minimum overlapping flat region for stats."
            )
            n = min(arr.size, compare_arr.size)
            stats = compare_arrays(arr.ravel()[:n], compare_arr.ravel()[:n])
        else:
            stats = compare_arrays(arr, compare_arr)
        print(f"\nComparison ({key} vs {comp_key}), record {args.record}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_dir = os.path.join(repo_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = args.out or os.path.join(plot_dir, "tfrecord_inspect.png")

    if compare_arr is not None and compare_arr.shape == arr.shape:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(arr, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title(f"A: {os.path.basename(args.path)}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(compare_arr, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title(f"B: {os.path.basename(args.compare)}")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        diff = arr - compare_arr
        im2 = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="coolwarm")
        axes[2].set_title("A − B")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"{key} — record {args.record}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
