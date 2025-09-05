# day5_ae_primer.py
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Utilities ----------------
def make_time(fs_hz: int, duration_s: float):
    N = int(round(fs_hz * duration_s))
    n = np.arange(N, dtype=np.int64)
    t = n / float(fs_hz)
    return t, n

def ae_pulse(fs, L, f0=80_000.0, decay_ms=0.3, phase=0.0, scale=1.0):
    """Decaying sinusoid; zero-mean and energy-normalized then scaled."""
    n = np.arange(L)
    t = n / fs
    tau = decay_ms / 1000.0
    p = np.exp(-t/tau) * np.sin(2*np.pi*f0*t + phase)
    p -= p.mean()
    p /= (np.linalg.norm(p) + 1e-12)
    return scale * p

def add_impulsive_noise(x, n_spikes=5, spike_amp=2.0, rng=None):
    """Add rare large spikes (electrical clicks, etc.)."""
    if rng is None: rng = np.random.default_rng(0)
    N = x.shape[-1]
    idx = rng.integers(0, N, size=n_spikes)
    x[..., idx] += spike_amp * (2*rng.random(n_spikes)-1.0)
    return x

def estimate_noise_power(x, fs, pre_ms=0.5):
    """Estimate noise power from a short pre-event region (first pre_ms)."""
    pre = int((pre_ms/1000.0)*fs)
    pre = min(pre, x.shape[-1])
    seg = x[..., :pre]
    return float(np.mean(seg*seg))

def cell_power(x, cell_size=500):
    """Compute mean-square power per non-overlapping cell along last axis."""
    N = x.shape[-1]
    n_full = N // cell_size
    used = n_full * cell_size
    if n_full == 0:
        return np.empty(0), np.empty(0)
    cells = x[..., :used].reshape(*x.shape[:-1], n_full, cell_size)
    p = np.mean(cells*cells, axis=-1)
    t_cell = (np.arange(n_full) + 0.5) * (cell_size/1_000_000.0)  # at fs=1 MHz, seconds
    return p, t_cell

# ---------------- Simulation ----------------
def simulate_multisensor_ae(
    fs=1_000_000, duration_ms=2.0, n_sensors=8,
    sensor_pos_m=None, source_pos_m=2.3, wave_speed_mps=5000.0,
    coupling=None, noise_sigma=0.2, f0_khz=80.0, decay_ms=0.25,
    events=None, seed=0
):
    """
    Returns (t, X) where X has shape [n_sensors, N].
    events: list of dicts with {'t_ms': float, 'amp': float}, burst start times at the source.
    """
    rng = np.random.default_rng(seed)
    dur_s = duration_ms/1000.0
    t, _ = make_time(fs, dur_s)
    N = len(t)
    if sensor_pos_m is None:
        # 8 sensors roughly 0..3.5 m
        sensor_pos_m = np.linspace(0.0, 3.5, n_sensors)
    if coupling is None:
        # default good coupling near 1.0 with small variation
        coupling = 0.85 + 0.3*rng.random(n_sensors)
    if events is None:
        events = [{"t_ms":0.6, "amp":0.9},
                  {"t_ms":1.1, "amp":1.0},
                  {"t_ms":1.6, "amp":0.8}]

    X = noise_sigma * rng.standard_normal((n_sensors, N))  # base Gaussian noise

    # add impulsive noise sparsely (optional)
    for i in range(n_sensors):
        add_impulsive_noise(X[i], n_spikes=3, spike_amp=1.5, rng=rng)

    # build one canonical pulse shape (L ≈ 0.4 ms at 1 MHz -> 400 samples)
    L = int(max(80, round(decay_ms/1000.0*fs*1.2)))  # slightly longer than tau
    base_pulse = ae_pulse(fs, L, f0=f0_khz*1000.0, decay_ms=decay_ms, phase=0.0, scale=1.0)

    # add each event with per-sensor delay and coupling
    for ev in events:
        t0 = ev["t_ms"]/1000.0
        amp0 = ev["amp"]
        for i, x in enumerate(X):
            d = abs(sensor_pos_m[i] - source_pos_m)  # meters
            delay_s = d / wave_speed_mps
            start_idx = int(round((t0 + delay_s) * fs))
            if start_idx < N:
                put = base_pulse * (amp0 * coupling[i])
                end_idx = min(N, start_idx + len(put))
                x[start_idx:end_idx] += put[:(end_idx-start_idx)]

    return t, X, np.asarray(sensor_pos_m), np.asarray(coupling)

# ---------------- Demo & Plots ----------------
def demo():
    fs = 1_000_000
    t, X, sensor_pos, coupling = simulate_multisensor_ae(
        fs=fs, duration_ms=2.0, n_sensors=8,
        source_pos_m=2.3, wave_speed_mps=5000.0,
        coupling=[0.95,0.9,0.85,0.8, 0.75,0.7,0.6,0.5],  # make late sensors a bit worse
        noise_sigma=0.20, f0_khz=80.0, decay_ms=0.25
    )

    # Pick one channel to inspect in time
    ch = 4  # like your S5
    print(f"Inspecting sensor {ch} at pos {sensor_pos[ch]:.2f} m, coupling {coupling[ch]:.2f}")
    # Show 0.4 ms around first event
    t_ms = t*1e3
    mask = (t_ms >= 0.4) & (t_ms <= 0.9)

    plt.figure(figsize=(9,3))
    plt.plot(t_ms[mask], X[ch, mask])
    plt.title(f"Sensor {ch}: AE bursts in time (zoom)")
    plt.xlabel("Time (ms)"); plt.ylabel("Amplitude")
    plt.tight_layout(); plt.show()

    # Compute SNR per sensor using early 0.5 ms as noise estimate
    snr_db = []
    noiseP = []
    sigP = []
    for i in range(X.shape[0]):
        Pn = estimate_noise_power(X[i], fs, pre_ms=0.5)
        Pt = float(np.mean(X[i]*X[i]))
        noiseP.append(Pn); sigP.append(Pt)
        snr_db.append(10*np.log10(max(Pt-Pn, 1e-12)/max(Pn, 1e-12)))
    snr_db = np.array(snr_db)

    # Cell power timelines for all sensors
    cp, tc = cell_power(X, cell_size=500)  # shape [sensors, n_cells]
    # plot cell power for 4 sensors
    show = [4,5,6,7]
    plt.figure(figsize=(10,4))
    for s in show:
        plt.plot(tc*1e3, cp[s], label=f"S{s} (SNR≈{snr_db[s]:.1f} dB)")
    plt.title("Cell mean power (500 samples ≙ 0.5 ms) — 4 sensors")
    plt.xlabel("Time (ms)"); plt.ylabel("Power"); plt.legend()
    plt.tight_layout(); plt.show()

    # Time-of-arrival vs distance (first event only)
    # crude pick: max in first 1.2 ms of cell power per sensor
    i_end = np.searchsorted(tc, 1.2e-3)   # 1.2 ms in seconds
    toa_cells = np.argmax(cp[:, :i_end], axis=1)  # index in cell units
    toa_sec = (toa_cells + 0.5) * 500 / fs
    plt.figure(figsize=(6,4))
    plt.plot(sensor_pos, toa_sec*1e3, marker='o')
    plt.title("First-event time-of-arrival vs sensor distance")
    plt.xlabel("Sensor position (m)"); plt.ylabel("TOA (ms)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    demo()
