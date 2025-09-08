# learn_day13.py
# A compact, hands-on lesson using your file. No SciPy needed.
import numpy as np, os, time
from day11_memmap_reader import WFSInterleavedReader
from numpy.lib.stride_tricks import sliding_window_view

# ---- Your paths & basic config ----
WFS_PATH = r"D:\Pipeline RUL Data\B.wfs"
SENSOR   = 4
FS       = 1_000_000     # Hz
CELL     = 500           # samples per cell (0.5 ms)
HEADER   = 2
DTYPE    = np.dtype("<i2")
NCH      = 8

# CFAR knobs
T, G = 50, 4
PFA  = 1e-4
Q    = 0.70  # OS-CFAR quantile

def iter_cells_seconds(path, seconds, sensor=SENSOR):
    rdr = WFSInterleavedReader(path, dtype=DTYPE, n_channels=NCH, header_bytes=HEADER, fs=FS)
    need_cells = int(seconds / (CELL/FS))
    have = 0; buf = []
    for _, block in rdr.iter_cell_power(cell_size=CELL, seconds_per_chunk=seconds):
        s = block[sensor]
        take = min(need_cells - have, s.size)
        buf.append(s[:take]); have += take
        if have >= need_cells: break
    return np.concatenate(buf)

# ---- Part 1: what is a cell? (noise stats) ----
def lesson_cells():
    print("\n[1] Cell power basics")
    # simulate noise-only cell-power: mean of squares over L samples
    L = CELL; rng = np.random.default_rng(0)
    sim = (rng.standard_normal((200000, L))**2).mean(axis=1)
    print(f"Sim noise cells: mean≈{sim.mean():.3f}, std≈{sim.std():.3f} (unitless)")
    print("  (Theory: mean≈1, std≈sqrt(2/L)≈{:.3f})".format(np.sqrt(2/L)))
    # Load 5 s of your data, estimate scale vs unit model
    real = iter_cells_seconds(WFS_PATH, 5.0)
    scale = real.mean()  # use mean as rough scale to relate to unit model
    print(f"Your data (5 s, S{SENSOR}): mean≈{real.mean():.3e}, std≈{real.std():.3e}")
    print("Takeaway: cell-power ≈ average of L squared samples; variance shrinks as 1/L.")

# ---- Part 2: CA-CFAR vs OS-CFAR intuition on one contaminated window ----
def alpha_ca(pfa, L, K, N=300000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L*K, size=N)/(L*K)  # mean of K training cells
    R = X / Y
    return float(np.quantile(R, 1.0-pfa))

def alpha_os(pfa, L, K, q, N=300000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L, size=(N, K))/L
    qv = np.quantile(Y, q, axis=1)
    R = X / qv
    return float(np.quantile(R, 1.0-pfa))

def lesson_contamination():
    print("\n[2] Why thresholds spike near bursts (contamination) & how OS helps")
    L = CELL; K = 2*T
    a_ca = alpha_ca(PFA, L, K)
    a_os = alpha_os(PFA, L, K, Q)
    print(f"Calibrated alphas: CA≈{a_ca:.3f}, OS(Q={Q})≈{a_os:.3f}")

    rng = np.random.default_rng(1)
    train = rng.chisquare(df=L, size=(K,))/L
    mean_clean = train.mean()
    q_clean = np.quantile(train, Q)
    # contaminate 20% of training with a 'burst' x4 power
    idx = rng.choice(K, size=int(0.2*K), replace=False)
    train2 = train.copy(); train2[idx] *= 4.0
    mean_cont = train2.mean()
    q_cont   = np.quantile(train2, Q)
    print(f"Clean training:   mean={mean_clean:.3f}, q{int(Q*100)}={q_clean:.3f}")
    print(f"Contaminated 20%: mean={mean_cont:.3f}, q{int(Q*100)}={q_cont:.3f}")
    print("CA threshold scales with mean → big rise; OS uses a quantile → much smaller rise.")

# ---- Part 3: Compare CA vs OS on a QUIET slice of your data ----
def cfar_ca_1d(x, L, T, G, alpha):
    W = T + G
    k = np.arange(-W, W+1)
    ker = ((np.abs(k) > G) & (np.abs(k) <= G+T)).astype(float)
    s = np.convolve(x, ker, mode="same"); c = np.convolve(np.ones_like(x), ker, mode="same")
    c = np.maximum(c, 1.0); thr = alpha * (s/c)
    valid = (c >= 0.8*(2*T))
    det = (x > thr) & valid
    return thr, det, valid

def cfar_os_1d(x, L, T, G, alpha, q):
    W = T + G
    win = sliding_window_view(x, 2*W+1)         # (N-2W, 2W+1)
    idx = np.arange(2*W+1); mask = (np.abs(idx - W) > G)
    train = win[:, mask]                        # (N-2W, 2T)
    qi = int(np.floor(q*(2*T-1)))
    part = np.partition(train, qi, axis=1)
    qv = part[:, qi]
    thr_core = alpha * qv
    thr = np.full_like(x, np.nan, float)
    det = np.zeros_like(x, bool)
    centers = np.arange(W, len(x)-W)
    thr[centers] = thr_core
    det[centers] = x[centers] > thr_core
    return thr, det, np.isfinite(thr)

def lesson_quiet_slice(seconds=30.0):
    print(f"\n[3] Quiet-slice PFA check (≈{seconds:.0f}s from your file)")
    x = iter_cells_seconds(WFS_PATH, seconds)  # first N seconds; good enough for now
    L = CELL; K = 2*T
    a_ca = alpha_ca(PFA, L, K); a_os = alpha_os(PFA, L, K, Q)
    thr_ca, det_ca, val_ca = cfar_ca_1d(x, L, T, G, a_ca)
    thr_os, det_os, val_os = cfar_os_1d(x, L, T, G, a_os, Q)
    pfa_ca = det_ca[val_ca].mean() if val_ca.any() else np.nan
    pfa_os = det_os[val_os].mean() if val_os.any() else np.nan
    print(f"CA: hits={det_ca.sum()} / valid={val_ca.sum()}  → hit fraction≈{pfa_ca:.3e}")
    print(f"OS: hits={det_os.sum()} / valid={val_os.sum()}  → hit fraction≈{pfa_os:.3e}")
    print("If these are near your target PFA on calm data, calibration makes sense;")
    print("on busy data, hit fraction will be much higher (not a 'PFA' anymore).")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    t0 = time.time()
    lesson_cells()
    lesson_contamination()
    lesson_quiet_slice(30.0)
    print("\nDone in {:.1f}s".format(time.time()-t0))
