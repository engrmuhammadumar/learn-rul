# day11_reader_cfar_run.py
import os, numpy as np
from datetime import datetime
from day11_memmap_reader import WFSInterleavedReader
from cfar_core import StreamingCFAR  # from Day 10

# ----------- CONFIG (edit here) -----------

WFS_PATH = r"D:\Pipeline RUL Data\B.wfs"
PIPE_NAME = "B"                  # just for filenames/columns
N_CHANNELS = 8
DTYPE = np.dtype("<i2")          # little-endian int16
HEADER_BYTES = 2
FS = 1_000_000                   # 1 MHz
CELL_SIZE = 500                  # 0.5 ms per cell

# CFAR params (same as Day 8/9/10 examples)
# CFAR params
T_TRAIN = 50
G_GUARD = 4
PFA     = 1e-4
USE_OS_CFAR = True     # <— add this
Q_QUANTILE = 0.70
                 # guard cells per side

# Streaming sizes
SECONDS_PER_CHUNK = 10.0         # how much raw time to pull per reader chunk
SECONDS_LIMIT = 60.0             # set to None for FULL RUN (≈ 1.9e7 cells total)
SENSORS_TO_RUN = [4,5,6,7]       # pick any subset, or list(range(8))
OUT_DIR = "cfar_outputs"
# -----------------------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rdr = WFSInterleavedReader(
        path=WFS_PATH,
        dtype=DTYPE,
        n_channels=N_CHANNELS,
        header_bytes=HEADER_BYTES,
        fs=FS,
    )
    print(rdr.info())

    # Create one streaming CFAR per sensor
from cfar_core import StreamingCFAR               # already there
from os_cfar_streaming import StreamingOSCFAR     # NEW

streamers = {}
for ch in SENSORS_TO_RUN:
    if USE_OS_CFAR:
        streamers[ch] = StreamingOSCFAR(
            L_cell=CELL_SIZE, T=T_TRAIN, G=G_GUARD, pfa=PFA, q=Q_QUANTILE,
            chunk_size_cells=int(SECONDS_PER_CHUNK * FS / CELL_SIZE),
            checkpoint_path=None
        )
    else:
        streamers[ch] = StreamingCFAR(
            L_cell=CELL_SIZE, T=T_TRAIN, G=G_GUARD, pfa=PFA,
            chunk_size_cells=int(SECONDS_PER_CHUNK * FS / CELL_SIZE),
            checkpoint_path=None
        )


    # Open one CSV per sensor
    writers = {}
    for ch in SENSORS_TO_RUN:
        fn = os.path.join(
            OUT_DIR,
            f"{PIPE_NAME}_S{ch}_cfar_pfa{PFA:g}_T{T_TRAIN}_G{G_GUARD}.csv"
        )
        f = open(fn, "w", buffering=1)
        f.write("pipe,sensor,cell_idx,time_s,det,thr\n")
        writers[ch] = f
        print("Writing ->", fn)

    cell_dt = CELL_SIZE / FS
    max_cells = None if SECONDS_LIMIT is None else int(SECONDS_LIMIT / cell_dt)

    # Iterate cell-power blocks
    for start_idx, block in rdr.iter_cell_power(
        cell_size=CELL_SIZE,
        seconds_per_chunk=SECONDS_PER_CHUNK
    ):
        n_cells = block.shape[1]

        # Early-exit if we've reached the limit
        if (max_cells is not None) and (start_idx >= max_cells):
            break

        # If the last chunk crosses the limit, trim it
        if (max_cells is not None) and (start_idx + n_cells > max_cells):
            n_cells = max_cells - start_idx
            block = block[:, :n_cells]

        for ch in SENSORS_TO_RUN:
            # Feed this channel's cell-power slice into its streamer
            cp_slice = block[ch]  # 1-D length = n_cells
            for glob_start, det_emit, thr_emit in streamers[ch].process_iter([cp_slice]):
                # Build time and indices for emitted region
                idxs  = glob_start + np.arange(det_emit.size, dtype=np.int64)
                times = (idxs + 0.5) * cell_dt

                # (Optional) keep column_stack, but keep all numeric (no placeholder)
                out = np.column_stack([
                    np.full_like(idxs, fill_value=ch, dtype=np.int64),  # sensor
                    idxs.astype(np.int64),                              # cell_idx
                    times.astype(np.float64),                           # time_s
                    det_emit.astype(np.int64),                          # det
                    thr_emit.astype(np.float64),                        # thr
                ])

                f = writers[ch]
                for row in out:
                    # pipe,sensor,cell_idx,time_s,det,thr
                    f.write(f"{PIPE_NAME},{int(row[0])},{int(row[1])},{row[2]:.6f},{int(row[3])},{row[4]:.6e}\n")

    # Close files
    for f in writers.values():
        f.close()
    print("Done.")


if __name__ == "__main__":
    main()
