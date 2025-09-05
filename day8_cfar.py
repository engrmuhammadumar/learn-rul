# day8_cfar.py
import math, numpy as np


# -------------------- Alpha (threshold factor) --------------------
def alpha_from_f_quantile(pfa: float, L: int, K: int):
    """
    Compute alpha so that P(X/Y > alpha | H0) = pfa, with X~(σ²/L)χ²_L, Y~(σ²/(KL))χ²_{KL}.
    Prefer SciPy's exact F-quantile; otherwise use a quick Monte Carlo fallback.
    """
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa must be in (0,1)")
    try:
        from scipy.stats import f
        return float(f.ppf(1.0 - pfa, L, K * L))
    except Exception:
        rng = np.random.default_rng(123)
        N = 400_000
        X = rng.normal(size=(N, L)) ** 2
        Y = rng.normal(size=(N, K * L)) ** 2
        Xbar = X.mean(axis=1)
        Ybar = Y.mean(axis=1)
        R = Xbar / Ybar
        q = np.quantile(R, 1.0 - pfa)
        return float(q)


# -------------------- Vectorized CA-CFAR --------------------
def cfar_kernel(T: int, G: int):
    Lk = 2 * (T + G) + 1
    k = np.arange(-(T + G), (T + G) + 1)
    ker = np.where((np.abs(k) > G) & (np.abs(k) <= G + T), 1.0, 0.0)
    return ker


def ca_cfar(cell_power: np.ndarray, L: int, T: int, G: int, pfa: float):
    x = np.asarray(cell_power, float)
    K = 2 * T
    ker = cfar_kernel(T, G)

    train_sum = np.convolve(x, ker, mode="same")
    train_cnt = np.convolve(np.ones_like(x), ker, mode="same")
    train_cnt = np.maximum(train_cnt, 1.0)
    local_mean = train_sum / train_cnt

    alpha = alpha_from_f_quantile(pfa, L=L, K=K)
    thr = alpha * local_mean
    det = x > thr

    valid = train_cnt >= (0.8 * K)
    det &= valid

    return thr, det, local_mean, alpha


# -------------------- Simulation --------------------
def simulate_stream(fs=1_000_000, duration_ms=50.0, L=500, bursts=()):
    dur_s = duration_ms / 1000.0
    N = int(fs * dur_s)
    rng = np.random.default_rng(7)
    sigma = 0.2
    x = sigma * rng.standard_normal(N)

    for b in bursts:
        t0 = int((b.get("t_ms", 10.0) / 1000.0) * fs)
        Lb = int(b.get("len_samp", 400))
        amp = float(b.get("amp", 1.0))
        x[t0:t0+Lb] += amp

    n_cells = N // L
    used = n_cells * L
    cells = x[:used].reshape(n_cells, L)
    cp = (cells * cells).mean(axis=1)
    tc = (np.arange(n_cells) + 0.5) * (L / fs)
    return cp, tc


# -------------------- Validation --------------------
def validate_noise_pfa():
    fs = 1_000_000
    L = 500
    T, G = 20, 2
    pfa = 1e-3

    cp, tc = simulate_stream(fs, duration_ms=200.0, L=L, bursts=())
    thr, det, mu, alpha = ca_cfar(cp, L=L, T=T, G=G, pfa=pfa)
    emp = det.mean()
    print(f"[Noise validation] Target PFA={pfa:g}, empirical≈{emp:.3e}, alpha={alpha:.4f}")

    # Plot noise-only validation
    plot_series(tc, cp, thr, det, title="Noise-only validation")
    return emp


def demo_with_bursts():
    fs = 1_000_000
    L = 500
    T, G = 20, 2
    pfa = 1e-3

    bursts = [
        {"t_ms": 10.0, "amp": 0.7, "len_samp": 400},
        {"t_ms": 22.0, "amp": 1.0, "len_samp": 500},
        {"t_ms": 35.0, "amp": 1.3, "len_samp": 600},
    ]
    cp, tc = simulate_stream(fs, duration_ms=50.0, L=L, bursts=bursts)
    thr, det, mu, alpha = ca_cfar(cp, L=L, T=T, G=G, pfa=pfa)

    n_det = int(det.sum())
    first_idxs = np.where(det)[0][:5]
    print(f"[Bursts demo] alpha={alpha:.4f}, detections={n_det}, first idxs={first_idxs}")

    # Plot burst demo
    plot_series(tc, cp, thr, det, title="Burst demo")


# -------------------- Plotting --------------------
def plot_series(tc, cp, thr, det, title="CA-CFAR"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(tc * 1e3, cp, label="cell power")
    plt.plot(tc * 1e3, thr, label="threshold")
    if np.any(det):
        plt.scatter(tc[det] * 1e3, cp[det], s=12, label="detections")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------- Main --------------------
if __name__ == "__main__":
    validate_noise_pfa()
    demo_with_bursts()
