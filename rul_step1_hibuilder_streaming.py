# rul_step1_hibuilder_streaming.py
# Memory-safe Step 1 of the paper:
# stream-read WFS -> cell energies (500 samples) -> CA-CFAR -> AE events -> cumulative HI
# Paper defaults: Pfa=1e-7, cell size=500, guard=10, train=20, SR=1 MHz.

import os, pathlib, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_DIR = r"D:\Pipeline RUL Data"
FILE_CONFIGS = {
    # file_name : (channels, channel_index_to_use)
    "Test20190731-104527-983.wfs": (1, 0),
    "Test20190731-142924-815.wfs": (2, 0),
    "Test20190731-172412-110.wfs": (8, 0),
    "Test20190731-200233-899.wfs": (8, 0),
}
HEADER_SKIP = 4096
DTYPE = ">i4"       # big-endian int32, from your inspector v2
SR = 1_000_000      # 1 MHz (paper)
# CFAR (paper)
PFA = 1e-7
CELL_SIZE = 500     # samples per cell (0.5 ms at 1 MHz)
N_GUARD = 10        # cells
N_TRAIN = 20        # cells on each side
# streaming chunk size in FRAMES (a "frame" = one sample across all channels)
CHUNK_FRAMES = 2_000_000
# ----------------------------

def file_num_frames(path, header_skip, dtype, n_ch):
    dt = np.dtype(dtype)
    total_bytes = os.path.getsize(path) - header_skip
    if total_bytes < 0:
        return 0
    elems = total_bytes // dt.itemsize
    frames = elems // n_ch
    return frames

def stream_channel(path, header_skip, dtype, n_ch, ch_idx, frames_per_chunk):
    """Yield 1-D numpy arrays of channel samples, chunk by chunk (float64)."""
    dt = np.dtype(dtype)
    itemsize = dt.itemsize
    with open(path, "rb") as f:
        f.seek(header_skip, 0)
        # compute total frames
        total_frames = file_num_frames(path, header_skip, dtype, n_ch)
        frames_read = 0
        while frames_read < total_frames:
            to_read = min(frames_per_chunk, total_frames - frames_read)
            # read to_read * n_ch elements
            count = to_read * n_ch
            raw = np.fromfile(f, dtype=dt, count=count)
            # safety
            if raw.size < count:
                # partial last read
                frames = raw.size // n_ch
                raw = raw[: frames * n_ch]
                to_read = frames
            if raw.size == 0:
                break
            frames_read += to_read
            if n_ch > 1:
                raw = raw.reshape(-1, n_ch)[:, ch_idx]
            # convert to float64 for energy math
            yield raw.astype(np.float64)

def cell_energies_stream(path, header_skip, dtype, n_ch, ch_idx, cell_size, frames_per_chunk):
    """
    Stream the file and accumulate sum of squares per 500-sample cell.
    Returns energy array (float32) of length n_cells.
    """
    buf = np.empty(0, dtype=np.float64)
    energies = []
    total_samples = 0
    for chunk in stream_channel(path, header_skip, dtype, n_ch, ch_idx, frames_per_chunk):
        if chunk.size == 0: 
            continue
        # prepend leftover
        if buf.size:
            chunk = np.concatenate([buf, chunk], axis=0)
            buf = np.empty(0, dtype=np.float64)

        # take full cells
        n_full = (chunk.size // cell_size) * cell_size
        if n_full > 0:
            cells = chunk[:n_full].reshape(-1, cell_size)
            # energy per cell = sum of squares
            e = np.sum(cells * cells, axis=1, dtype=np.float64)
            energies.append(e.astype(np.float32))
            total_samples += n_full

        # keep leftover
        rem = chunk.size - n_full
        if rem > 0:
            buf = chunk[-rem:]

    # drop any leftover < full cell (paper uses complete cells)
    if energies:
        E = np.concatenate(energies, axis=0)
    else:
        E = np.zeros(0, dtype=np.float32)

    return E

def cfar_alpha_ca(pfa, N):
    # CA-CFAR threshold factor alpha
    return N * (pfa**(-1.0 / N) - 1.0)

def ca_cfar_prefixsum(energy, n_guard, n_train, pfa):
    """
    CA-CFAR using prefix sums (O(N)):
    For cell i, noise = mean(left train + right train), excluding guard and CUT.
    """
    E = energy.astype(np.float64)
    N = 2 * n_train
    alpha = cfar_alpha_ca(pfa, N)
    n = E.size
    det = np.zeros(n, dtype=bool)
    # prefix sums for fast range sums
    P = np.zeros(n + 1, dtype=np.float64)
    P[1:] = np.cumsum(E)

    # valid range to evaluate
    start = n_train + n_guard
    end   = n - (n_train + n_guard) - 1
    if end < start:
        return det, None, alpha

    # sliding indices
    for i in range(start, end + 1):
        # left training window [i - n_guard - n_train, i - n_guard)
        L0 = i - n_guard - n_train
        L1 = i - n_guard
        # right training window (i + n_guard, i + n_guard + n_train]
        R0 = i + n_guard + 1
        R1 = i + n_guard + n_train + 1

        sum_left  = P[L1] - P[L0]
        sum_right = P[R1] - P[R0]
        Pn = (sum_left + sum_right) / N
        Th = alpha * Pn
        if E[i] > Th:
            det[i] = True

    return det, None, alpha

def flags_to_events(flags):
    f = flags.astype(np.int8)
    if f.size == 0:
        return []
    diff = np.diff(np.r_[0, f, 0])
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))

def cumulative_hi(flags, cell_size, sr):
    events = flags_to_events(flags)
    n_cells = flags.size
    HI = np.zeros(n_cells, dtype=np.int32)
    c = 0
    for (s,e) in events:
        c += 1
        HI[e:] += 1
    t = (np.arange(n_cells) + 1) * (cell_size / sr)
    return t, HI, events

def process_one(path, n_ch, ch_idx):
    print(f"\n[INFO] Processing {os.path.basename(path)} | ch={n_ch} use={ch_idx}")
    # 1) energies
    energy = cell_energies_stream(path, HEADER_SKIP, DTYPE, n_ch, ch_idx, CELL_SIZE, CHUNK_FRAMES)
    print(f"[INFO] cells: {energy.size}  (approx duration: {energy.size * CELL_SIZE / SR:.1f} s)")

    # 2) CFAR on energies
    flags, _, alpha = ca_cfar_prefixsum(energy, N_GUARD, N_TRAIN, PFA)
    print(f"[INFO] alpha={alpha:.3f}, hits={flags.sum()} cells")

    # 3) cumulative HI
    t, HI, events = cumulative_hi(flags, CELL_SIZE, SR)
    print(f"[INFO] events={len(events)}")

    return energy, flags, (t, HI, events)

def main():
    out_dir = "hi_outputs_streaming"
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    summary = []

    for fname, (n_ch, ch_idx) in FILE_CONFIGS.items():
        fp = os.path.join(DATA_DIR, fname)
        energy, flags, (t, HI, events) = process_one(fp, n_ch, ch_idx)

        stem = pathlib.Path(fname).stem

        # save HI
        df = pd.DataFrame({"t_s": t, "HI": HI})
        df.to_csv(os.path.join(out_dir, f"{stem}_HI.csv"), index=False)

        # quick plots
        plt.figure(figsize=(10,3))
        plt.plot(t, HI)
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative AE-hit (HI)")
        plt.title(stem)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_HI.png"), dpi=150)
        plt.close()

        # (optional) save energies & flags for debugging
        np.save(os.path.join(out_dir, f"{stem}_energy.npy"), energy)
        np.save(os.path.join(out_dir, f"{stem}_flags.npy"), flags)

        summary.append(dict(
            file=fname,
            n_cells=int(energy.size),
            n_events=int(len(events)),
            hi_csv=f"{stem}_HI.csv",
            hi_png=f"{stem}_HI.png",
        ))

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "HI_summary.csv"), index=False)
    print("\nDone. See:", out_dir)

if __name__ == "__main__":
    main()
