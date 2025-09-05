# day6_noise_stats.py
import math, numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist

# ---------- Math helpers ----------
def chi2_ppf_approx(df: int, p: float) -> float:
    """
    Wilson–Hilferty approximation for chi-square quantile.
    Good for df >= ~30; perfect for our L=500 cells.
    """
    if not (0.0 < p < 1.0): raise ValueError("p must be in (0,1)")
    z = NormalDist().inv_cdf(p)  # inverse standard normal CDF
    k = float(df)
    a = 2.0/(9.0*k)
    return k * (1.0 - a + z*math.sqrt(a))**3

def fixed_sigma_threshold(sigma: float, L: int, pfa: float) -> float:
    """T such that P( mean_power > T ) = pfa for Gaussian noise with known sigma."""
    q = 1.0 - pfa
    chi2_q = chi2_ppf_approx(L, q)
    return (sigma*sigma/L) * chi2_q

# ---------- Experiments ----------
def monte_carlo_pfa(sigma: float, L: int, T: float, trials: int = 200_000) -> float:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, sigma, size=(trials, L))
    pwr = np.mean(x*x, axis=1)
    return float(np.mean(pwr > T))

def histogram_vs_theory(sigma: float, L: int, Ncells: int = 200_000):
    """
    Show histogram of mean-power and overlay a Gamma fit:
      mean = sigma^2, var = 2*sigma^4/L.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, sigma, size=(Ncells, L))
    mp = np.mean(x*x, axis=1)

    mu = sigma*sigma
    var = 2.0 * (sigma**4) / L
    # Gamma parameterization: shape k, scale θ with mean = kθ, var = kθ^2
    k = (mu*mu)/var
    theta = var/mu

    # Plot
    plt.figure(figsize=(8,4))
    plt.hist(mp, bins=120, density=True, alpha=0.6, label="Empirical")
    # crude Gamma pdf via sampling (no SciPy): draw many gamma samples and KDE-like plot
    # instead we approximate with Normal since L large (CLT) just to show alignment
    from math import sqrt, pi, exp
    xs = np.linspace(mu - 6*math.sqrt(var), mu + 6*math.sqrt(var), 400)
    pdf_norm = (1.0/np.sqrt(2*pi*var)) * np.exp(-0.5*((xs-mu)**2)/var)
    plt.plot(xs, pdf_norm, label=f"Normal approx (μ={mu:.3g}, σ={math.sqrt(var):.3g})")
    plt.axvline(mu, color='k', linestyle='--', alpha=0.6, label="σ² (mean)")
    plt.title(f"Cell mean-power (L={L}) under Gaussian noise")
    plt.xlabel("mean power"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.show()

def pfa_table():
    sig = 0.20
    L_list = [100, 250, 500]
    pfa_list = [1e-2, 1e-3, 1e-4]
    rows = []
    for L in L_list:
        for pfa in pfa_list:
            T = fixed_sigma_threshold(sig, L, pfa)
            emp = monte_carlo_pfa(sig, L, T, trials=100_000)
            margin_db = 10.0*math.log10(T/(sig*sig) + 1e-30)  # dB above noise power
            rows.append((L, pfa, T, emp, margin_db))
            print(f"L={L:4d}  PFA={pfa:1.0e}  T={T:.6f}  empPFA={emp:.3e}  (T is {margin_db:+.2f} dB over σ²)")
    return rows

def visualize_threshold(sigma=0.2, L=500, pfa=1e-3):
    T = fixed_sigma_threshold(sigma, L, pfa)
    # Monte Carlo one long stream as contiguous cells (like your pipeline)
    fs = 1_000_000
    duration_s = 2.0
    N = int(fs*duration_s)
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, sigma, size=N)

    # cell mean-power timeline
    n_cells = N//L
    used = n_cells*L
    cells = x[:used].reshape(n_cells, L)
    mp = (cells*cells).mean(axis=1)
    t_cell = (np.arange(n_cells)+0.5)*L/fs

    # plot
    plt.figure(figsize=(10,4))
    plt.plot(t_cell, mp, marker='o', ms=3, lw=1)
    plt.axhline(T, color='r', linestyle='--', label=f"Threshold for PFA≈{pfa:g}")
    plt.title(f"Noise-only mean power across cells (L={L}, σ={sigma})")
    plt.xlabel("Time (s)"); plt.ylabel("mean power"); plt.legend(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sigma = 0.20
    L = 500
    print("== Histogram vs theory ==")
    histogram_vs_theory(sigma, L, Ncells=120_000)

    print("\n== PFA table (analytic approx vs Monte Carlo) ==")
    pfa_table()

    print("\n== Timeline view with threshold ==")
    visualize_threshold(sigma=sigma, L=L, pfa=1e-3)
