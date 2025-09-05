# day4_power_spectra.py
import numpy as np
import matplotlib.pyplot as plt

# ---------- utilities ----------
def timebase(fs_hz: int, duration_s: float):
    N = int(round(fs_hz * duration_s))
    n = np.arange(N, dtype=np.int64)
    t = n / float(fs_hz)
    return t, n

def sine(A=1.0, f=50.0, fs=1000, dur=1.0, phi=0.0):
    t, _ = timebase(fs, dur)
    x = A * np.sin(2*np.pi*f*t + phi)
    return t, x

def ae_pulse(fs, L, f0=80_000.0, decay_ms=0.3, scale=1.0, seed=0):
    """Decaying sinusoid 'AE-like' normalized, then scaled."""
    n = np.arange(L)
    t = n / fs
    tau = decay_ms/1000.0
    p = np.exp(-t/tau) * np.sin(2*np.pi*f0*t)
    p -= p.mean()
    p /= (np.linalg.norm(p) + 1e-12)
    return scale * p

# ---------- Parseval demo ----------
def parseval_demo():
    fs, dur = 1000, 1.0
    t, _ = timebase(fs, dur)
    # multi-tone + noise
    x = (1.2*np.sin(2*np.pi*5*t)
         +0.7*np.sin(2*np.pi*12*t+0.3)
         +0.5*np.sin(2*np.pi*40*t+1.0))
    x += 0.2*np.random.default_rng(0).standard_normal(len(t))

    # time-domain energy
    E_time = float(np.sum(x*x))

    # frequency-domain energy via full FFT (Parseval)
    X = np.fft.fft(x)
    E_freq = float(np.sum(np.abs(X)**2) / len(x))

    print(f"[Parseval] E_time={E_time:.6f}  E_freq={E_freq:.6f}  |diff|={abs(E_time-E_freq):.3e}")

    # plot spectrum (one-sided, magnitude)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    Xr = np.fft.rfft(x * np.hanning(len(x)))
    mag = np.abs(Xr) / (len(x)/2)  # rough amplitude scaling
    plt.figure()
    plt.plot(freqs, mag)
    plt.title("One-sided magnitude spectrum (Hann window)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude (approx)")
    plt.tight_layout(); plt.show()

# ---------- Welch PSD (NumPy-only) ----------
def welch_psd(x, fs, nperseg=1024, noverlap=None, window="hann"):
    """
    Simple Welch PSD (one-sided) in units of power/Hz.
    Returns freqs (Hz), psd (power/Hz).
    """
    x = np.asarray(x, float)
    N = len(x)
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    if step <= 0 or nperseg > N:
        raise ValueError("Bad nperseg/noverlap for signal length.")

    # window
    if window == "hann":
        w = np.hanning(nperseg)
    else:
        w = np.ones(nperseg)
    U = (w**2).mean()  # window power normalization

    # segment indices
    starts = np.arange(0, N - nperseg + 1, step, dtype=int)
    K = len(starts)
    psd_accum = None

    for s in starts:
        seg = x[s:s+nperseg]
        segw = seg * w
        X = np.fft.rfft(segw)
        # two-sided periodogram scaled to one-sided PSD (power/Hz)
        Pxx = (np.abs(X)**2) / (fs * nperseg * U)
        # one-sided: double all bins except DC and Nyquist (if exists)
        if nperseg % 2 == 0:
            # even nperseg includes Nyquist bin
            Pxx[1:-1] *= 2.0
        else:
            Pxx[1:] *= 2.0
        if psd_accum is None:
            psd_accum = Pxx
        else:
            psd_accum += Pxx

    psd = psd_accum / K
    freqs = np.fft.rfftfreq(nperseg, d=1/fs)
    return freqs, psd

def psd_power_check():
    fs, dur = 2000, 2.0
    t, _ = timebase(fs, dur)
    x = 1.0*np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t+0.1)
    x += 0.3*np.random.default_rng(1).standard_normal(len(t))

    P_time = float(np.mean(x*x))  # average power in time
    f, Sxx = welch_psd(x, fs, nperseg=512, noverlap=256, window="hann")
    df = f[1]-f[0]
    P_psd = float(np.trapz(Sxx, f))  # area under PSD ≈ average power

    print(f"[PSD] time-domain power={P_time:.6f}  PSD integral={P_psd:.6f}  |diff|={abs(P_time-P_psd):.3e}")

    plt.figure()
    plt.semilogy(f, Sxx)
    plt.title("Welch PSD (semilogy)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power/Hz")
    plt.tight_layout(); plt.show()

# ---------- Cell power demo (CFAR intuition) ----------
def cell_power_demo():
    fs = 1_000_000        # 1 MHz like your data
    dur = 0.02            # 20 ms (keep short)
    t, _ = timebase(fs, dur)
    x = 0.2*np.random.default_rng(2).standard_normal(len(t))

    # Insert a few AE-like pulses
    L = 400  # 0.4 ms pulse
    starts = [3000, 8000, 13000]
    for s in starts:
        x[s:s+L] += ae_pulse(fs, L, f0=80_000, decay_ms=0.25, scale=0.8)

    # Cell power (Lcell=500 samples ≙ 0.5 ms)
    Lcell = 500
    n_cells = len(x)//Lcell
    used = n_cells*Lcell
    cells = x[:used].reshape(n_cells, Lcell)
    cell_pwr = (cells*cells).mean(axis=1)

    # Plot time segment + cell power timeline
    tt = np.arange(used)/fs
    tc = (np.arange(n_cells)+0.5)*Lcell/fs  # center time of each cell

    fig, axs = plt.subplots(2,1, figsize=(10,6), sharex=False)
    axs[0].plot(tt*1e3, x[:used])
    axs[0].set_title("Signal with AE-like bursts (20 ms @ 1 MHz)")
    axs[0].set_xlabel("Time (ms)"); axs[0].set_ylabel("Amp")

    axs[1].plot(tc*1e3, cell_pwr, marker='o')
    axs[1].set_title("Cell mean power (500-sample cells ≙ 0.5 ms)")
    axs[1].set_xlabel("Time (ms)"); axs[1].set_ylabel("Power")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    parseval_demo()
    psd_power_check()
    cell_power_demo()
