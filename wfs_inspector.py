# wfs_inspector.py
# Purpose: understand unknown `.wfs` files (binary layout, dtype, header, channels).

import os
import sys
import math
import struct
import pathlib
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG: EDIT THIS ----------
DATA_DIR = r"D:\Pipeline RUL Data"   # <= put your folder path here
FILE_GLOB = "*.wfs"
# Candidate assumptions to try:
HEADER_SKIPS = [0, 512, 4096]        # bytes to skip before samples
DTYPES      = ["<i2","<i4","<f4",">i2",">i4",">f4"]  # little/big int16,int32,float32
N_CHANNELS  = [1, 2, 4, 8]           # interleaved channels (1 = mono)
MAX_SAMPLES_FOR_PREVIEW = 1_000_000  # to keep things snappy
PLOT_PREVIEW_SECONDS = 0.01          # duration for preview plot (seconds)
SR_GUESS = 1_000_000                 # 1 MHz sampling rate from the paper (you can adjust)

OUT_SUMMARY_CSV = "wfs_inspector_summary.csv"
OUT_PREVIEW_DIR = "wfs_previews"
# --------------------------------------

pathlib.Path(OUT_PREVIEW_DIR).mkdir(exist_ok=True)

def is_mostly_ascii(b: bytes, thresh=0.85) -> bool:
    printable = sum([32 <= c <= 126 or c in (9,10,13) for c in b])
    return (printable / max(1, len(b))) >= thresh

def head_bytes(path: str, n=512) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)

def file_size(path: str) -> int:
    return os.path.getsize(path)

def try_read_payload(path: str, header_skip: int, dtype: str, n_channels: int, max_samples: int):
    sz = file_size(path)
    if header_skip >= sz:
        return None, "header_skip >= file size"

    byte_per_sample = np.dtype(dtype).itemsize
    payload_bytes = sz - header_skip
    if payload_bytes <= 0:
        return None, "no payload"

    # number of elements total
    n_total = payload_bytes // byte_per_sample
    if n_total == 0:
        return None, "payload too small for dtype"

    # to keep things fast, cap how much we load
    n_to_read = min(n_total, max_samples * n_channels)
    with open(path, "rb") as f:
        f.seek(header_skip, 0)
        arr = np.fromfile(f, dtype=dtype, count=n_to_read)

    # If channels>1, deinterleave: shape (-1, n_channels)
    if n_channels > 1:
        n_frames = (arr.size // n_channels)
        if n_frames == 0:
            return None, "not enough data for channel split"
        arr = arr[: n_frames * n_channels].reshape(n_frames, n_channels)
    else:
        arr = arr.reshape(-1, 1)  # (n,1) for uniform handling

    return arr, None

def score_signal(arr: np.ndarray) -> Dict[str, Any]:
    """
    Heuristic score: penalize arrays that are constant or saturated,
    prefer reasonable variance and few extreme spikes.
    """
    # work on first channel
    x = arr[:,0].astype(np.float64)
    n = x.size
    if n == 0:
        return dict(score=-1e9, reason="empty", **{})

    mean = float(x.mean())
    std  = float(x.std(ddof=1)) if n > 1 else 0.0
    vmin = float(x.min())
    vmax = float(x.max())
    ptp  = vmax - vmin

    # fraction near zero (too high might indicate wrong dtype scaling)
    frac_near_zero = float((np.abs(x) < 1e-6).mean()) if x.dtype.kind == 'f' else float((x == 0).mean())

    # fraction at dtype limits (saturation)
    if x.dtype.kind in ('i','u'):
        info = np.iinfo(arr.dtype)
        frac_sat = float(((x <= info.min) | (x >= info.max)).mean())
    else:
        frac_sat = 0.0

    # simple outlier check
    if std > 0:
        z = (x - mean) / std
        frac_abs_z_gt_8 = float((np.abs(z) > 8).mean())
    else:
        frac_abs_z_gt_8 = 0.0

    # scoring: target non-zero std, limited saturation & outliers
    score = 0.0
    score += + np.clip(std, 0, 1e6) / 1000.0
    score += - 5.0 * frac_near_zero
    score += - 20.0 * frac_sat
    score += - 2.0 * frac_abs_z_gt_8
    score += + 0.001 * np.log10(1.0 + ptp) if ptp > 0 else -10.0

    return dict(score=score, mean=mean, std=std, vmin=vmin, vmax=vmax,
                frac_near_zero=frac_near_zero, frac_saturation=frac_sat,
                frac_abs_z_gt_8=frac_abs_z_gt_8, n_samples=int(n))

def preview_plot(path: str, arr: np.ndarray, sr: float, seconds: float, save_to: str):
    n = arr.shape[0]
    nshow = int(seconds * sr)
    nshow = min(n, max(10, nshow))
    t = np.arange(nshow) / sr
    x = arr[:nshow, 0]

    plt.figure(figsize=(9,3))
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Preview: {os.path.basename(path)} (first {seconds:.3f}s)")
    plt.tight_layout()
    plt.savefig(save_to, dpi=160)
    plt.close()

def inspect_file(path: str) -> List[Dict[str, Any]]:
    results = []
    # peek at header
    hb = head_bytes(path, 256)
    looks_ascii = is_mostly_ascii(hb, 0.80)
    hex_head = hb[:64].hex(" ")

    for header_skip in HEADER_SKIPS:
        for dt in DTYPES:
            for ch in N_CHANNELS:
                arr, err = try_read_payload(path, header_skip, dt, ch, MAX_SAMPLES_FOR_PREVIEW)
                if err:
                    results.append(dict(
                        file=path, header_skip=header_skip, dtype=dt, channels=ch,
                        score=-1e9, error=err, looks_ascii_header=looks_ascii, hex_head=hex_head))
                    continue
                stats = score_signal(arr)
                rec = dict(file=path, header_skip=header_skip, dtype=dt, channels=ch,
                           looks_ascii_header=looks_ascii, hex_head=hex_head, **stats)
                results.append(rec)

    # pick best by score and save plot
    if results:
        best = max(results, key=lambda r: r["score"])
        if best["score"] > -1e9:
            arr, _ = try_read_payload(path, best["header_skip"], best["dtype"], best["channels"], MAX_SAMPLES_FOR_PREVIEW)
            out_png = os.path.join(OUT_PREVIEW_DIR,
                                   f"{pathlib.Path(path).stem}__hs{best['header_skip']}__{best['dtype']}__ch{best['channels']}.png")
            try:
                preview_plot(path, arr, SR_GUESS, PLOT_PREVIEW_SECONDS, out_png)
            except Exception as e:
                print(f"[WARN] preview plot failed for {path}: {e}")

    return results

def main():
    folder = pathlib.Path(DATA_DIR)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    files = sorted(str(p) for p in folder.glob(FILE_GLOB))
    if not files:
        print(f"No files matching {FILE_GLOB} in {folder}")
        sys.exit(0)

    all_rows = []
    for fp in files:
        print(f"Inspecting: {fp}")
        rows = inspect_file(fp)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).sort_values(["file","score"], ascending=[True, False])
    # Keep top-5 candidates per file for readability
    df_top = df.groupby("file").head(5)
    df_top.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\nSaved summary: {OUT_SUMMARY_CSV}")
    print(f"Preview plots (if any): {OUT_PREVIEW_DIR}\\*.png")
    print("\nTop candidates per file:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_top[["file","score","header_skip","dtype","channels","n_samples","mean","std","vmin","vmax","frac_near_zero","frac_saturation"]])

if __name__ == "__main__":
    main()
