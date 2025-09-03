# calibrate_os_alpha_empirical.py
# Empirically calibrate OS-CFAR alpha from calm seconds and cache it for StreamingOSCFAR.

import os, json, numpy as np
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view
from day11_memmap_reader import WFSInterleavedReader

# ---------- CONFIG ----------
WFS_PATH = r"D:\Pipeline RUL Data\B.wfs"
SENSORS_FOR_CALM = [4,5,6,7]   # use these to pick calm seconds
CALIBRATE_SENSOR = 4           # compute alpha from this sensor (do per-sensor if you want)
FS = 1_000_000
CELL = 500
DTYPE = np.dtype("<i2"); HEADER = 2; NCH = 8

SCAN_SECONDS = 3600.0          # scan 1 hour; increase if needed
TOP_K_SECONDS = 120            # use this many calm seconds for calibration

# OS-CFAR knobs (must match your run)
T, G = 50, 4
Q = 0.70
TARGET_PFA = 1e-4

ALPHA_CACHE_PATH = "os_cfar_alpha_cache.json"  # StreamingOSCFAR reads this by default
# ---------------------------

def find_calm_seconds(rdr, sensors, scan_seconds, cell):
    cells_per_sec = rdr.fs // cell
    need_cells = int(scan_seconds * cells_per_sec)
    buckets = {s: defaultdict(list) for s in sensors}
    seen = 0
    for start, block in rdr.iter_cell_power(cell_size=cell, seconds_per_chunk=10.0):
        n = block.shape[1]
        if seen >= need_cells: break
        if seen + n > need_cells:
            n = need_cells - seen
            block = block[:, :n]
        secs = (start + np.arange(n)) // cells_per_sec
        uniq = np.unique(secs)
        for s in sensors:
            x = block[s]
            for u in uniq:
                m = (secs == u)
                buckets[s][int(u)].append(x[m])
        seen += n

    seconds = sorted(set().union(*[set(buckets[s].keys()) for s in sensors]))
    scored = []
    for sec in seconds:
        spreads = []
        ok = True
        for s in sensors:
            if sec not in buckets[s]:
                ok = False; break
            arr = np.concatenate(buckets[s][sec])
            if arr.size < 100: ok = False; break
            q1, q3 = np.percentile(arr, [25, 75])
            spreads.append(float(q3 - q1))
        if ok:
            scored.append((max(spreads), sec))
    scored.sort(key=lambda t: t[0])
    return [sec for _, sec in scored[:TOP_K_SECONDS]], buckets

def ratios_for_segments(segments, L, T, G, q):
    """Compute r = X / Qq(training) for each valid center, per segment (no cross-boundary leakage)."""
    W = T + G; K = 2*T
    qi = int(np.floor(q * (K - 1)))
    rs = []
    for seg in segments:
        x = np.asarray(seg, float)
        if x.size < 2*W+1: continue
        win = sliding_window_view(x, 2*W+1)       # (N-2W, 2W+1)
        idx = np.arange(2*W+1)
        mask = (np.abs(idx - W) > G)              # training only
        train = win[:, mask]                       # (N-2W, K)
        part = np.partition(train, qi, axis=1)
        qv = part[:, qi]                           # per-center quantile
        X = x[W:len(x)-W]                          # centers
        r = X / qv
        rs.append(r)
    if not rs: return np.empty((0,), float)
    return np.concatenate(rs)

def main():
    rdr = WFSInterleavedReader(WFS_PATH, dtype=DTYPE, n_channels=NCH, header_bytes=HEADER, fs=FS)
    calm_secs, buckets = find_calm_seconds(rdr, SENSORS_FOR_CALM, SCAN_SECONDS, CELL)
    if not calm_secs:
        print("No calm seconds found. Increase SCAN_SECONDS.")
        return
    print(f"Using {len(calm_secs)} calm seconds (first few): {calm_secs[:10]}")

    # build segments for the calibration sensor
    segs = [np.concatenate(buckets[CALIBRATE_SENSOR][sec]) for sec in calm_secs]
    r = ratios_for_segments(segs, L=CELL, T=T, G=G, q=Q)
    if r.size == 0:
        print("No valid centers in selected segments.")
        return

    alpha_emp = float(np.quantile(r, 1.0 - TARGET_PFA))
    print(f"Empirical OS-CFAR alpha for PFA={TARGET_PFA:g} (L={CELL}, T={T}, G={G}, q={Q}): {alpha_emp:.6f}")

    # write/merge cache
    key = f"L={CELL}|K={2*T}|q={Q}|pfa={TARGET_PFA}"
    cache = {}
    if os.path.exists(ALPHA_CACHE_PATH):
        try:
            with open(ALPHA_CACHE_PATH, "r") as f: cache = json.load(f)
        except Exception:
            cache = {}
    cache[key] = alpha_emp
    tmp = ALPHA_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f: json.dump(cache, f, indent=2)
    os.replace(tmp, ALPHA_CACHE_PATH)
    print(f"Wrote {ALPHA_CACHE_PATH} with key '{key}'. Your StreamingOSCFAR will pick this up automatically.")

if __name__ == "__main__":
    main()
