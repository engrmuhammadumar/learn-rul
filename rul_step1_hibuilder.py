# rul_step1_hibuilder.py
# Step 1 of the paper: read WFS -> CFAR hit detection -> cumulative HI
# Paper defaults: Pfa=1e-7, cell size=500 samples (0.5 ms at 1 MHz), 10 guard, 20 training.

import os, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ======= CONFIG (set from inspector v2 output) =======
DATA_DIR = r"D:\Pipeline RUL Data"
FILES = [
    "Test20190731-104527-983.wfs",
    "Test20190731-142924-815.wfs",
    "Test20190731-172412-110.wfs",
    "Test20190731-200233-899.wfs",
]
HEADER_SKIP = 4096
DTYPE = ">i4"        # big-endian int32
CHANNELS = 1        # set per file if needed (can be 1,2,4,8)
SR = 1_000_000      # 1 MHz as in the paper

# CFAR params (paper)
PFA = 1e-7
CELL_SIZE = 500     # samples
N_GUARD = 10        # cells
N_TRAIN = 20        # cells
# =====================================================

def read_wfs(path, header_skip=4096, dtype=">i4", n_ch=1):
    dt = np.dtype(dtype)
    with open(path, "rb") as f:
        f.seek(header_skip, 0)
        raw = np.fromfile(f, dtype=dt)
    if n_ch > 1:
        frames = raw.size // n_ch
        raw = raw[:frames*n_ch].reshape(frames, n_ch)
    else:
        raw = raw.reshape(-1,1)
    # Convert to float for processing
    return raw.astype(np.float64)

def ca_cfar_threshold_factor(pfa, N):
    # CA-CFAR threshold factor alpha for given Pfa and N training cells
    # alpha = N * (Pfa**(-1/N) - 1)
    return N * (pfa**(-1.0/N) - 1.0)

def ca_cfar_hits(x, cell_size=500, n_guard=10, n_train=20, pfa=1e-7):
    """
    Vectorized CA-CFAR over energy in non-overlapping cells.
    Returns: boolean array per cell indicating detection.
    """
    # Energy per cell (sum of squares)
    n = (x.size // cell_size) * cell_size
    x = x[:n]
    cells = x.reshape(-1, cell_size)
    energy = np.sum(cells**2, axis=1)

    N = n_train * 2
    alpha = ca_cfar_threshold_factor(pfa, N)
    # For each CUT index k, estimate noise from neighbor training cells excluding guard cells
    # We’ll do a simple moving average with exclusion.
    from numpy.lib.stride_tricks import sliding_window_view
    win = n_train + n_guard + 1 + n_guard + n_train
    if energy.size < win:
        return np.zeros(energy.shape[0], dtype=bool), energy, None

    # sliding window over energies
    E = sliding_window_view(energy, win)  # shape: (L, win)
    # indices to include as training: left n_train and right n_train
    mask = np.ones((win,), dtype=bool)
    mask[n_train:n_train+1+2*n_guard] = False  # CUT and guards removed
    train_vals = E[:, mask]                    # (L, 2*n_train)
    Pn = train_vals.mean(axis=1)               # noise power estimate
    Th = alpha * Pn

    # CUT indices aligned to E rows
    cut = energy[n_train+ n_guard : -(n_train + n_guard)]
    det = cut > Th
    # pad to original length
    det_full = np.zeros_like(energy, dtype=bool)
    det_full[n_train+ n_guard : -(n_train + n_guard)] = det
    return det_full, energy, Th

def hits_to_events(hit_flags):
    """
    Group consecutive 'True' cells as one AE event (as in the paper).
    Returns event_indices (start_cell, end_cell) and event_count.
    """
    flags = hit_flags.astype(np.int8)
    if flags.size == 0:
        return [], 0
    diff = np.diff(np.r_[0, flags, 0])
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    events = list(zip(starts, ends))
    return events, len(events)

def hi_from_events(hit_flags):
    """
    Cumulative HI = cumulative count of AE events over time.
    HI is defined on cell indices; we convert to time using SR and CELL_SIZE.
    """
    events, _ = hits_to_events(hit_flags)
    n_cells = hit_flags.size
    cum = np.zeros(n_cells, dtype=int)
    count = 0
    for (s,e) in events:
        count += 1
        cum[e:] += 1  # increment from the end of each event forward
    # time per cell endpoint:
    t = (np.arange(n_cells) + 1) * (CELL_SIZE / SR)
    return t, cum, events

def process_file(fp, channels=1):
    x = read_wfs(fp, HEADER_SKIP, DTYPE, channels)[:,0]  # take ch0 for now
    flags, energy, Th = ca_cfar_hits(x, CELL_SIZE, N_GUARD, N_TRAIN, PFA)
    t, HI, events = hi_from_events(flags)
    out = {
        "time_s": t,
        "HI": HI,
        "n_events": len(events),
        "n_cells": flags.size,
        "energy": energy,
        "flags": flags
    }
    return out

def main():
    out_dir = "hi_outputs"
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    summary = []
    for name in FILES:
        fp = os.path.join(DATA_DIR, name)
        print(f"Processing: {fp}")
        res = process_file(fp, CHANNELS)
        # Save HI to CSV
        df = pd.DataFrame({"t_s": res["time_s"], "HI": res["HI"]})
        csv_path = os.path.join(out_dir, f"{pathlib.Path(name).stem}_HI.csv")
        df.to_csv(csv_path, index=False)

        # Quick plot
        plt.figure(figsize=(9,3))
        plt.plot(res["time_s"], res["HI"])
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative AE-hit count (HI)")
        plt.title(f"HI: {name}")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{pathlib.Path(name).stem}_HI.png")
        plt.savefig(png_path, dpi=160)
        plt.close()

        summary.append(dict(file=name, n_cells=res["n_cells"], n_events=res["n_events"],
                            hi_csv=csv_path, hi_png=png_path))
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "HI_summary.csv"), index=False)
    print("Done. See folder:", out_dir)

if __name__ == "__main__":
    main()
