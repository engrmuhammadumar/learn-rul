# day11_os_cfar_demo.py
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from day11_memmap_reader import WFSInterleavedReader

# ---- CONFIG ----
WFS_PATH = r"D:\Pipeline RUL Data\B.wfs"
SENSOR = 4                  # try 4..7
FS = 1_000_000
CELL = 500
SECONDS = 60.0              # slice to analyze
T, G = 50, 4                # training, guard
Q = 0.7                     # use 70th percentile of training cells
PFA = 1e-4                  # target false alarm on noise
# ----------------

def alpha_os_cfar(pfa: float, L: int, K: int, q: float, N: int = 300_000) -> float:
    """
    Calibrate α so that P( X / Qq(training) > α | H0 ) = pfa,
    where X and each training cell ~ (1/L) * χ²_L  (unit-variance noise).
    """
    rng = np.random.default_rng(123)
    # CUT mean-power under H0
    X = rng.chisquare(df=L, size=N) / L
    # K training cell powers per trial
    Y = rng.chisquare(df=L, size=(N, K)) / L
    qv = np.quantile(Y, q, axis=1)
    R = X / qv
    return float(np.quantile(R, 1.0 - pfa))

def os_cfar_1d(x: np.ndarray, L: int, T: int, G: int, q: float, alpha: float):
    """
    Vectorized OS-CFAR on 1D cell-power array.
    Returns thr (same length as x, NaN at edges) and det (bool).
    """
    x = np.asarray(x, float)
    W = T + G
    win = sliding_window_view(x, window_shape=2*W+1)  # shape (N-2W, 2W+1)
    # mask: keep only training positions (exclude center and guards)
    idx = np.arange(2*W+1)
    keep = (np.abs(idx - W) > G)  # True for training, False for CUT and guards
    train = win[:, keep]          # shape (N-2W, K) where K=2T
    # robust noise estimate (quantile per position)
    qv = np.quantile(train, q, axis=1)
    thr_core = alpha * qv
    # map back to full-length arrays
    thr = np.full_like(x, np.nan, dtype=float)
    det = np.zeros_like(x, dtype=bool)
    center_idx = np.arange(W, len(x)-W)  # valid centers
    thr[center_idx] = thr_core
    det[center_idx] = x[center_idx] > thr_core
    return thr, det

def main():
    # 1) load ~SECONDS of cell power for one sensor
    rdr = WFSInterleavedReader(WFS_PATH, dtype=np.dtype("<i2"), n_channels=8, header_bytes=2, fs=FS)
    need_cells = int(SECONDS / (CELL/FS))
    have = 0
    cp = []
    for start, block in rdr.iter_cell_power(cell_size=CELL, seconds_per_chunk=SECONDS):
        cp.append(block[SENSOR])
        have += block.shape[1]
        if have >= need_cells: break
    cp = np.concatenate(cp)[:need_cells]
    print(f"Loaded {len(cp)} cells for sensor {SENSOR} (~{SECONDS:.1f}s)")

    # 2) calibrate α for OS-CFAR and run it
    K = 2*T
    alpha = alpha_os_cfar(PFA, L=CELL, K=K, q=Q, N=300_000)
    thr_os, det_os = os_cfar_1d(cp, L=CELL, T=T, G=G, q=Q, alpha=alpha)
    print(f"OS-CFAR: Q={Q}, PFA={PFA:g}, alpha≈{alpha:.4f}, hits={det_os.sum()} of {np.isfinite(thr_os).sum()} valid cells")

    # 3) (optional) compare to standard CA-CFAR on the same slice
    from cfar_core import ca_cfar_full
    thr_ca, det_ca, _, alpha_ca, valid = ca_cfar_full(cp, L=CELL, T=T, G=G, pfa=PFA)
    print(f"CA-CFAR: alpha≈{alpha_ca:.4f}, hits={int(det_ca.sum())} of {int(valid.sum())} valid cells")

    # 4) visualize threshold (OS-CFAR)
    t = (np.arange(len(cp)) + 0.5) * (CELL/FS)
    plt.figure(figsize=(10,3))
    plt.plot(t, thr_os, lw=1, label="OS-CFAR thr")
    plt.scatter(t[det_os], thr_os[det_os], s=8, label="detections")
    plt.xlabel("Time (s)"); plt.ylabel("Threshold"); plt.title(f"Sensor {SENSOR} — OS-CFAR")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
