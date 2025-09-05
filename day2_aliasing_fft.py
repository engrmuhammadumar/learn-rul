# day2_aliasing_fft.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Utilities ----------
def timebase(fs_hz: int, duration_s: float):
    N = int(round(fs_hz * duration_s))
    n = np.arange(N, dtype=np.int64)
    t = n / float(fs_hz)
    return t, n

def sine(A=1.0, f=50.0, fs=1000, dur=1.0, phi=0.0):
    t, _ = timebase(fs, dur)
    x = A * np.sin(2*np.pi*f*t + phi)
    return t, x

def alias_after_sampling(f_cont, fs):
    # bring into [0, fs)
    fprime = np.fmod(f_cont, fs)
    if fprime < 0:
        fprime += fs
    # fold into [0, fs/2]
    if fprime <= fs/2:
        return fprime
    return fs - fprime

def fft_mag(x, fs):
    # one-sided magnitude spectrum
    N = len(x)
    X = np.fft.rfft(x * np.hanning(N))  # light window to reduce leakage
    mag = np.abs(X) / (N/2)              # scale roughly to amplitude
    freqs = np.fft.rfftfreq(N, d=1/fs)
    return freqs, mag

def plot_time(t, x, title=""):
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if title: plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spectrum(freqs, mag, title=""):
    plt.figure()
    plt.plot(freqs, mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    if title: plt.title(title)
    plt.tight_layout()
    plt.show()

# ---------- Demos ----------
def demo_aliasing_basic():
    fs = 1000
    dur = 1.0
    true_f = 750.0  # > fs/2 (=500), will alias
    alias = alias_after_sampling(true_f, fs)
    print(f"[Basic] fs={fs} Hz, true f={true_f} Hz -> predicted alias={alias:.2f} Hz")

    t, x = sine(A=1.0, f=true_f, fs=fs, dur=dur)
    f, m = fft_mag(x, fs)
    pk = f[np.argmax(m)]
    print(f"[Basic] FFT peak ≈ {pk:.2f} Hz")

    # quick plots
    plot_time(t[:200], x[:200], f"Time (first 0.2 s) — tone {true_f} Hz @ fs={fs}")
    plot_spectrum(f, m, "Spectrum (expect a peak near alias)")

def demo_aliasing_table():
    fs = 1000
    tests = [100, 300, 510, 700, 900, 1200, 1490]  # diverse tones
    rows = []
    for ftrue in tests:
        alias = alias_after_sampling(ftrue, fs)
        rows.append({"f_true": ftrue, "f_alias_pred": alias})
    df = pd.DataFrame(rows)
    print("\nAliasing predictions (fs=1000 Hz):")
    print(df.to_string(index=False))

def demo_downsample_without_filter():
    fs = 2000
    dur = 1.0
    ftrue = 900.0
    t, x = sine(A=1.0, f=ftrue, fs=fs, dur=dur)

    # Downsample by M=4 (new fs=500) WITHOUT anti-alias filter
    M = 4
    x_ds = x[::M]
    fs_ds = fs // M
    t_ds = np.arange(len(x_ds))/fs_ds

    alias_pred = alias_after_sampling(ftrue, fs_ds)
    print(f"\n[Downsample] Start fs={fs}, f={ftrue} Hz -> after M={M} (fs'={fs_ds}) predicted alias={alias_pred:.2f} Hz")

    f1, m1 = fft_mag(x, fs)
    f2, m2 = fft_mag(x_ds, fs_ds)

    plot_spectrum(f1, m1, f"Spectrum @ fs={fs} Hz (true {ftrue} Hz)")
    plot_spectrum(f2, m2, f"Spectrum AFTER downsample to fs={fs_ds} Hz")

def demo_leakage_vs_onbin():
    fs = 1000
    dur = 1.0
    # Case A: exactly on a bin -> sharp peak
    f_onbin = 50.0            # since N=1000, bin spacing=1 Hz, 50 is exact
    # Case B: slightly off-bin -> leakage
    f_off = 50.7

    t, x1 = sine(1.0, f_onbin, fs, dur)
    t, x2 = sine(1.0, f_off,   fs, dur)

    f1, m1 = fft_mag(x1, fs)
    f2, m2 = fft_mag(x2, fs)

    plot_spectrum(f1, m1, f"On-bin {f_onbin} Hz — tall narrow spike")
    plot_spectrum(f2, m2, f"Off-bin {f_off} Hz — energy leaks to neighbors")

if __name__ == "__main__":
    demo_aliasing_basic()
    demo_aliasing_table()
    demo_downsample_without_filter()
    demo_leakage_vs_onbin()
