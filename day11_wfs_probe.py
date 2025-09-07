# day11_wfs_probe.py
import os, argparse, math
import numpy as np

DTYPES = [
    ("int16", np.dtype("<i2")),   # little-endian
    ("uint16", np.dtype("<u2")),
    ("int32", np.dtype("<i4")),
    ("float32", np.dtype("<f4")),
]
HEADER_CAND = [0, 2, 4, 8, 16, 32, 64, 128, 256]

def human(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def probe(path, n_channels=8, fs=1_000_000, expected_sec=None, max_rows=18):
    size = os.path.getsize(path)
    rows = []
    for name, dt in DTYPES:
        bps = dt.itemsize
        for hdr in HEADER_CAND:
            if hdr >= size: continue
            total_vals = (size - hdr) // bps
            # require exact channel alignment
            if total_vals % n_channels != 0:
                continue
            spc = total_vals // n_channels
            dur = spc / fs
            score = 0.0
            notes = []
            if expected_sec is not None:
                score += abs(dur - expected_sec)
            # prefer small headers
            score += 0.001 * hdr
            rows.append((score, name, "<", bps, hdr, int(spc), dur))
    rows.sort(key=lambda r: r[0])

    print(f"File: {path}  ({human(size)})")
    print("=== viable candidates (sorted) ===")
    for i, (_, name, endian, bps, hdr, spc, dur) in enumerate(rows[:max_rows]):
        print(f"[{i:2d}] dtype={name}, endian=little, header={hdr:3d} B, "
              f"samples/ch={spc:,}  duration={dur:.2f}s")

    # Recommend the first one (closest to expected / smallest header)
    if rows:
        _, name, endian, bps, hdr, spc, dur = rows[0]
        print("\n=== RECOMMENDED config.yml values ===")
        print(f'dtype: "{name}"')
        print(f'endianness: "little"')
        print(f"header_bytes: {hdr}")
        print(f"samples_per_channel: {spc:,}  (sanity: duration ~ {dur:.2f}s)")
    else:
        print("No clean alignment found. Try adding more header candidates or a different dtype list.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--fs", type=int, default=1_000_000)
    ap.add_argument("--expected_sec", type=float, default=None,
                    help="If you know ~duration (e.g., 6888.76), ranking improves.")
    args = ap.parse_args()
    probe(args.path, args.channels, args.fs, args.expected_sec)

if __name__ == "__main__":
    main()
