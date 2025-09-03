# day1_signals.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_time(fs_hz: int, duration_s: float):
    """Return time vector t (seconds) and sample indices n."""
    N = int(round(fs_hz * duration_s))
    n = np.arange(N, dtype=np.int64)
    t = n / float(fs_hz)
    return t, n

def sine(A=1.0, f=5.0, fs=1000, dur=1.0, phi=0.0):
    """Discrete-time sine x[n] = A*sin(2π f n/fs + phi)."""
    t, n = make_time(fs, dur)
    x = A * np.sin(2*np.pi*f*t + phi)
    return t, x

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def demo_single_sine():
    fs = 1000   # Hz
    dur = 1.0   # seconds
    A, f = 2.0, 5.0  # amplitude 2, freq 5 Hz
    t, x = sine(A=A, f=f, fs=fs, dur=dur, phi=0.0)
    print(f"Samples: {len(x)}, fs={fs} Hz, duration={t[-1]:.3f} s")
    print(f"RMS(x)={rms(x):.4f}  (theory A/sqrt(2)={A/np.sqrt(2):.4f})")

    plt.figure()
    plt.plot(t, x)
    plt.title(f"Single sine: A={A}, f={f} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def demo_multi_sine():
    fs, dur = 1000, 1.0
    t, x1 = sine(A=1.0, f=5, fs=fs, dur=dur)
    _, x2 = sine(A=0.6, f=12, fs=fs, dur=dur, phi=np.pi/4)
    _, x3 = sine(A=0.4, f=40, fs=fs, dur=dur, phi=np.pi/2)
    x = x1 + x2 + x3

    # Put into a tidy table for later habits
    df = pd.DataFrame({"t_s": t, "x": x, "x1_5Hz": x1, "x2_12Hz": x2, "x3_40Hz": x3})
    print(df.head())

    plt.figure()
    plt.plot(t, x, label="sum")
    plt.plot(t, x1, alpha=0.7, label="5 Hz")
    plt.plot(t, x2, alpha=0.7, label="12 Hz")
    plt.plot(t, x3, alpha=0.7, label="40 Hz")
    plt.legend()
    plt.title("Sum of three sinusoids")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def save_and_reload():
    fs, dur = 1000, 1.0
    t, x = sine(A=1.5, f=8, fs=fs, dur=dur, phi=0.2)
    df = pd.DataFrame({"t_s": t, "x": x})
    df.to_csv("day1_signal.csv", index=False)
    print("Wrote day1_signal.csv")

    df2 = pd.read_csv("day1_signal.csv")
    print("Reloaded:", df2.shape, "first rows:\n", df2.head())

if __name__ == "__main__":
    demo_single_sine()
    demo_multi_sine()
    save_and_reload()

