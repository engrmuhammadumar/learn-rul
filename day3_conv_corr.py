# day3_conv_corr.py
import numpy as np
import matplotlib.pyplot as plt

# ---------- utilities ----------
def make_time(fs, dur_s):
    N = int(round(fs*dur_s))
    n = np.arange(N)
    t = n / fs
    return t, n

def conv_naive(x, h):
    """Pure-Python convolution (valid for teaching; slow)."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    N = len(x); M = len(h)
    y = np.zeros(N+M-1, dtype=float)
    for n in range(N):
        y[n:n+M] += x[n] * h
    return y

def corr_naive(x, y):
    """Cross-correlation r_xy[m] = sum_n x[n] y[n+m]; full range."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.correlate(x, y, mode="full")  # numpy already does the right thing

def same_convolution(x, h):
    """Convolution with 'same' size output (centered)."""
    return np.convolve(x, h, mode="same")

# ---------- demo 1: convolution as filtering ----------
def demo_convolution_smoothing():
    fs, dur = 1000, 1.0
    t, _ = make_time(fs, dur)
    # signal: sin + noise
    x = np.sin(2*np.pi*5*t) + 0.5*np.random.randn(len(t))
    # a simple moving-average filter h[n] (boxcar of length L)
    L = 25
    h = np.ones(L)/L
    y = same_convolution(x, h)

    plt.figure()
    plt.plot(t, x, label="noisy input", alpha=0.6)
    plt.plot(t, y, label=f"MA conv, L={L}")
    plt.xlabel("Time (s)"); plt.ylabel("Amp"); plt.title("Convolution = Filtering")
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- demo 2: corr = conv with reversed template ----------
def demo_corr_equals_conv_reverse():
    fs, dur = 200, 1.0
    t, _ = make_time(fs, dur)
    # template: short pulse
    L = 41
    n = np.arange(L)
    templ = np.exp(-n/10.0) * np.sin(2*np.pi*10*n/fs)
    templ /= np.linalg.norm(templ) + 1e-12  # normalize energy

    # signal: zeros + template at an offset + small noise
    x = 0.05*np.random.randn(len(t))
    offset = 80
    x[offset:offset+L] += templ  # embed template

    # correlation (full)
    r = np.correlate(x, templ, mode="full")
    # convolution with time-reversed template
    h_rev = templ[::-1]
    c = np.convolve(x, h_rev, mode="full")

    # The two should be (nearly) identical (floating error aside)
    diff = np.max(np.abs(r - c))
    print(f"max|corr - conv(rev)| = {diff:.3e}")

    # Plot around the expected peak
    lags = np.arange(-len(templ)+1, len(x))
    plt.figure()
    plt.plot(lags, r, label="cross-corr r_xy[m]")
    plt.xlabel("Lag (samples)"); plt.ylabel("Score")
    plt.title("Correlation peak reveals alignment")
    plt.axvline(offset, color='k', linestyle='--', alpha=0.6, label="true lag")
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- demo 3: matched filter detection ----------
def make_ae_pulse(fs, L, f0=50e3, decay_ms=0.4):
    """
    AE-like pulse: decaying sinusoid at f0 (Hz), sampled at fs (Hz), length L samples.
    """
    n = np.arange(L)
    t = n / fs
    tau = decay_ms/1000.0
    pulse = np.exp(-t/tau) * np.sin(2*np.pi*f0*t)
    # zero-mean and energy normalize for a clean correlator
    pulse -= pulse.mean()
    pulse /= (np.linalg.norm(pulse) + 1e-12)
    return pulse

def matched_filter_detect():
    fs = 1_000_000  # 1 MHz like your data (we'll keep durations tiny for speed)
    dur = 0.02      # 20 ms record
    t, _ = make_time(fs, dur)
    N = len(t)

    # Build a template (AE-like)
    L = 400        # 400 samples @1 MHz = 0.4 ms
    templ = make_ae_pulse(fs, L, f0=80e3, decay_ms=0.25)

    # Make a signal with noise and ONE embedded pulse at unknown time
    rng = np.random.default_rng(0)
    x = 0.2 * rng.standard_normal(N)  # noise
    start = 6000                      # where pulse starts
    x[start:start+L] += 0.8 * templ   # embed scaled pulse

    # Matched filter: correlate with template
    rfull = np.correlate(x, templ, mode="full")
    lags = np.arange(-len(templ)+1, len(x))   # alignment of template start vs signal start

    # Convert to "same"-length score aligned to signal index
    score = np.convolve(x, templ[::-1], mode="same")  # same as correlation centered
    # Estimate noise-only std of the score using regions far from the pulse
    margin = 2000
    noise_zone = np.r_[score[:start-margin], score[start+L+margin:]]
    sigma = noise_zone.std() + 1e-9
    thr = 5.0 * sigma  # ~"5-sigma" threshold (informal today)

    # Find detections (indexes where score crosses threshold)
    hits = np.where(score > thr)[0]

    # Plots
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
    axs[0].plot(t*1e3, x)
    axs[0].set_title("Signal (20 ms @ 1 MHz) with one AE-like pulse")
    axs[0].set_ylabel("Amp")
    axs[0].set_xlabel("Time (ms)")

    axs[1].plot(lags, rfull)
    axs[1].set_title("Full cross-correlation (for intuition)")
    axs[1].set_ylabel("corr score"); axs[1].set_xlabel("Lag (samples)")
    axs[1].axvline(start, color='k', linestyle='--', alpha=0.6, label="true start")
    axs[1].legend()

    axs[2].plot(np.arange(N)/fs*1e3, score, label="matched filter score")
    axs[2].axhline(thr, color='r', linestyle='--', label=f"thr ≈ 5σ ({thr:.3f})")
    if hits.size:
        axs[2].scatter(hits/fs*1e3, score[hits], s=12, label="detections")
    axs[2].set_title("Matched filter (same-mode conv with reversed template)")
    axs[2].set_xlabel("Time (ms)"); axs[2].set_ylabel("score")
    axs[2].legend()
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    demo_convolution_smoothing()
    demo_corr_equals_conv_reverse()
    matched_filter_detect()
