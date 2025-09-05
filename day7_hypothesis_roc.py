# day7_hypothesis_roc.py
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist


# ---------- Utilities ----------
def ae_pulse(fs, L, f0=80_000.0, decay_ms=0.25):
    """Unit-energy decaying sinusoid of length L (samples)."""
    n = np.arange(L)
    t = n / fs
    tau = decay_ms / 1000.0
    p = np.exp(-t / tau) * np.sin(2 * np.pi * f0 * t)
    p -= p.mean()
    p /= (np.linalg.norm(p) + 1e-12)
    return p


def make_trials_H0_H1(ntrials, L, sigma, amp, pulse):
    """
    Generate stats under H0 (noise) and H1 (signal+noise).
    Energy detector uses the L-sample window (cell).
    Matched filter correlates with the (unit-energy) pulse, zero-padded to L.
    """
    rng = np.random.default_rng(0)
    # window where the pulse sits (aligned at start for simplicity)
    x0 = rng.normal(0.0, sigma, size=(ntrials, L))
    x1 = x0 + amp * pulse  # add same aligned pulse in each H1 trial

    # Energy detector (mean-square in the cell)
    eng0 = np.mean(x0**2, axis=1)
    eng1 = np.mean(x1**2, axis=1)

    # Matched filter (template has unit energy)
    mf0 = (x0 @ pulse)
    mf1 = (x1 @ pulse)

    return eng0, eng1, mf0, mf1


def roc_from_samples(h0, h1, npts=200):
    """
    Build an ROC by sweeping thresholds over combined samples.
    Returns arrays (PFA, PD), sorted by ascending PFA.
    Uses open-interval quantiles to avoid exact PFA==0 or 1.
    """
    all_vals = np.concatenate([h0, h1])
    # avoid exact 0 and 1 quantiles to reduce PFA==0 or 1
    q = np.linspace(0.0, 1.0, npts + 2)[1:-1]  # (0,1) open interval
    thr = np.quantile(all_vals, q)

    PFA = (h0[:, None] > thr[None, :]).mean(axis=0)
    PD = (h1[:, None] > thr[None, :]).mean(axis=0)
    order = np.argsort(PFA)
    return PFA[order], PD[order]


def roc_theory_matched(pfa, snr_mf, eps=1e-12):
    """
    Theoretical ROC for matched filter with known signal in white Gaussian noise:
    PD = Q(Q^{-1}(PFA) - sqrt(SNR_MF)), where SNR_MF = (amp/sigma)^2.

    Clamps PFA to (eps, 1-eps) to keep NormalDist().inv_cdf valid.
    """
    pfa = np.asarray(pfa, dtype=float)
    pfa = np.clip(pfa, eps, 1.0 - eps)

    # Q^{-1}(p) = z such that Q(z)=p; here Q(z)=1-CDF(z)
    invQ = lambda p: NormalDist().inv_cdf(1.0 - p)
    Q = lambda z: 1.0 - NormalDist().cdf(z)

    z = np.array([invQ(p) for p in pfa], dtype=float)
    return np.array([Q(zz - math.sqrt(snr_mf)) for zz in z], dtype=float)


def snr_params_from_cell_snr_db(cell_snr_db, L, sigma):
    """
    Choose amplitude 'amp' for a unit-energy pulse so that
    SNR_cell (per L-sample cell) equals the target (in dB).
    Then SNR_MF = (amp/sigma)^2 = L * SNR_cell.
    """
    snr_cell = 10 ** (cell_snr_db / 10.0)
    amp = math.sqrt(L * (sigma**2) * snr_cell)
    snr_mf = (amp / sigma) ** 2
    return amp, snr_mf


# ---------- Demo ----------
def main():
    fs = 1_000_000
    L = 500               # 500-sample "cell" (0.5 ms @ 1 MHz)
    sigma = 0.20          # noise std (same as Day 6)
    pulse_len = 400       # AE pulse ~0.4 ms within the cell
    pulse = np.zeros(L, dtype=float)
    pulse[:pulse_len] = ae_pulse(fs, pulse_len, f0=80_000.0, decay_ms=0.25)  # unit-energy overall

    ntrials = 10000       # Monte Carlo trials per ROC

    # Try a few cell-SNR levels (dB)
    cell_snr_db_list = [-10, -5, 0]

    plt.figure(figsize=(8.5, 6))
    for csdb in cell_snr_db_list:
        amp, snr_mf = snr_params_from_cell_snr_db(csdb, L, sigma)
        eng0, eng1, mf0, mf1 = make_trials_H0_H1(ntrials, L, sigma, amp, pulse)

        pfa_e, pd_e = roc_from_samples(eng0, eng1)
        pfa_m, pd_m = roc_from_samples(mf0, mf1)

        # theory for matched filter (internally clips)
        pd_m_theory = roc_theory_matched(pfa_m, snr_mf)

        # For plotting on log-x, avoid log(0) purely for display:
        pfa_plot_e = np.clip(pfa_e, 1e-6, 1.0)
        pfa_plot_m = np.clip(pfa_m, 1e-6, 1.0)

        plt.plot(pfa_plot_e, pd_e, label=f"Energy (cell)  SNRcell={csdb} dB")
        plt.plot(pfa_plot_m, pd_m, '--', label=f"Matched (MC)  SNRmf={10*math.log10(snr_mf):.1f} dB")
        plt.plot(np.clip(pfa_m, 1e-6, 1.0), pd_m_theory, ':', label="Matched (theory)")

    plt.xscale("log")
    plt.xlabel("PFA (false alarm)")
    plt.ylabel("PD (detection)")
    plt.title(f"ROC: Energy vs Matched Filter  (L={L}, σ={sigma})")
    plt.grid(True, which="both", ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print one operating point example (e.g., PFA ~ 1e-3)
    target_pfa = 1e-3
    amp, snr_mf = snr_params_from_cell_snr_db(0, L, sigma)
    eng0, eng1, mf0, mf1 = make_trials_H0_H1(ntrials, L, sigma, amp, pulse)

    pfa_e, pd_e = roc_from_samples(eng0, eng1)
    idx_e = int(np.argmin(np.abs(pfa_e - target_pfa)))

    pfa_m, pd_m = roc_from_samples(mf0, mf1)
    idx_m = int(np.argmin(np.abs(pfa_m - target_pfa)))

    theory_at_idx_m = roc_theory_matched(np.array([pfa_m[idx_m]]), snr_mf)[0]

    print(f"At PFA≈{pfa_e[idx_e]:.2e}: Energy PD≈{pd_e[idx_e]:.3f}")
    print(f"At PFA≈{pfa_m[idx_m]:.2e}: Matched PD≈{pd_m[idx_m]:.3f} (theory≈{theory_at_idx_m:.3f})")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
