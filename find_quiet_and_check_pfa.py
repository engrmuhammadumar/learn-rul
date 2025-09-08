# find_quiet_and_check_pfa.py
# Find the calmest seconds (robust spread across S4..S7), then measure empirical PFA there.
import numpy as np
from collections import defaultdict
from day11_memmap_reader import WFSInterleavedReader
from numpy.lib.stride_tricks import sliding_window_view

# ---- Config you may tweak ----
WFS_PATH = r"D:\Pipeline RUL Data\B.wfs"
SENSORS   = [4,5,6,7]     # use these to select calm seconds
CHECK_SENSOR = 4          # run CFAR PFA check on this sensor
FS = 1_000_000
CELL = 500                # 0.5 ms per cell
DTYPE = np.dtype("<i2")
HEADER = 2
NCH = 8

# scan how much?
SCAN_SECONDS = 600.0      # analyze first 10 minutes (adjust as you like)
TOP_K_SECONDS = 30        # how many calm seconds to evaluate CFAR on

# CFAR knobs (same as your experiments)
T, G = 50, 4
PFA  = 1e-4
Q    = 0.70  # OS-CFAR quantile
# --------------------------------

def alpha_ca(pfa, L, K, N=400_000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L*K, size=N)/(L*K)  # mean of K training
    R = X / Y
    return float(np.quantile(R, 1.0-pfa))

def alpha_os(pfa, L, K, q, N=400_000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L, size=(N, K))/L
    qv = np.quantile(Y, q, axis=1)
    R = X / qv
    return float(np.quantile(R, 1.0-pfa))

def cfar_ca_1d(x, L, T, G, alpha):
    W = T + G
    k = np.arange(-W, W+1)
    ker = ((np.abs(k) > G) & (np.abs(k) <= G+T)).astype(float)
    s = np.convolve(x, ker, mode="same"); c = np.convolve(np.ones_like(x), ker, mode="same")
    c = np.maximum(c, 1.0); thr = alpha * (s/c)
    valid = (c >= 0.8*(2*T))
    det = (x > thr) & valid
    return det, valid

def cfar_os_1d(x, L, T, G, alpha, q):
    W = T + G
    win = sliding_window_view(x, 2*W+1)             # (N-2W, 2W+1)
    idx = np.arange(2*W+1); mask = (np.abs(idx - W) > G)
    train = win[:, mask]                            # (N-2W, 2T)
    qi = int(np.floor(q*(2*T-1)))
    part = np.partition(train, qi, axis=1)
    qv = part[:, qi]
    thr_core = alpha * qv
    det = np.zeros_like(x, bool)
    valid = np.zeros_like(x, bool)
    centers = np.arange(W, len(x)-W)
    det[centers] = x[centers] > thr_core
    valid[centers] = True
    return det, valid

def main():
    rdr = WFSInterleavedReader(WFS_PATH, dtype=DTYPE, n_channels=NCH, header_bytes=HEADER, fs=FS)
    cells_per_sec = FS // CELL
    need_cells = int(SCAN_SECONDS * cells_per_sec)

    # Collect cell-power per second per sensor
    buckets = {s: defaultdict(list) for s in SENSORS}  # buckets[sensor][sec_idx] -> list of cp
    seen_cells = 0
    for start_idx, block in rdr.iter_cell_power(cell_size=CELL, seconds_per_chunk=10.0):
        n = block.shape[1]
        if seen_cells >= need_cells: break
        if seen_cells + n > need_cells:
            n = need_cells - seen_cells
            block = block[:, :n]
        # global seconds for these cells
        secs = (start_idx + np.arange(n)) // cells_per_sec
        # append
        for s in SENSORS:
            x = block[s]
            # group by sec; few unique per chunk — do it vectorized-ish
            uniq = np.unique(secs)
            for u in uniq:
                mask = (secs == u)
                buckets[s][int(u)].append(x[mask])
        seen_cells += n

    # Compute per-second robust spread (IQR) per sensor, then take the max across sensors
    seconds = sorted(set().union(*[set(buckets[s].keys()) for s in SENSORS]))
    scores = []
    for sec in seconds:
        spreads = []
        ok = True
        for s in SENSORS:
            if sec not in buckets[s]:
                ok = False; break
            arr = np.concatenate(buckets[s][sec])
            if arr.size < 100: ok = False; break
            q1, q3 = np.percentile(arr, [25, 75])
            spreads.append(float(q3 - q1))
        if ok:
            score = max(spreads)  # calm only if all sensors are calm
            scores.append((score, sec))
    if not scores:
        print("No seconds scored; try increasing SCAN_SECONDS.")
        return

    scores.sort(key=lambda t: t[0])
    quiet_secs = [sec for _, sec in scores[:TOP_K_SECONDS]]
    print(f"Picked {len(quiet_secs)} calm seconds (min IQR across S4–S7). "
          f"Earliest few: {quiet_secs[:10]}")

    # Build a 1-D array of cell power from those seconds for CHECK_SENSOR
    cp = []
    total_cells = 0
    # reuse buckets collected above for speed
    for sec in quiet_secs:
        seg = np.concatenate(buckets[CHECK_SENSOR][sec])
        cp.append(seg)
        total_cells += seg.size
    x = np.concatenate(cp)
    print(f"CFAR check on sensor {CHECK_SENSOR}: {len(x)} cells "
          f"(~{len(x)/cells_per_sec:.1f} s of calm data)")

    # Calibrate alphas and measure empirical PFA on the calm cells
    L = CELL; K = 2*T
    a_ca = alpha_ca(PFA, L, K); a_os = alpha_os(PFA, L, K, Q)
    det_ca, val_ca = cfar_ca_1d(x, L, T, G, a_ca)
    det_os, val_os = cfar_os_1d(x, L, T, G, a_os, Q)
    pfa_ca = det_ca[val_ca].mean() if val_ca.any() else np.nan
    pfa_os = det_os[val_os].mean() if val_os.any() else np.nan

    print(f"\nEmpirical PFA on *calm* seconds (target {PFA:g}):")
    print(f"  CA-CFAR : {pfa_ca:.3e}   (alpha≈{a_ca:.3f}, T={T}, G={G})")
    print(f"  OS-CFAR : {pfa_os:.3e}   (alpha≈{a_os:.3f}, T={T}, G={G}, Q={Q})")
    print("\nNotes:")
    print("- If these are near target, calibration is correct.")
    print("- On busy seconds the hit fraction will be much higher by design.")
    print("- Increase TOP_K_SECONDS for tighter estimates, or SCAN_SECONDS to search longer.")
    print("- If PFA is still high on calm seconds, raise T/G or tighten PFA, or try SO-CA.")

if __name__ == "__main__":
    main()
