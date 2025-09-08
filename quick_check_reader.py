# quick_check_reader.py
from day11_memmap_reader import WFSInterleavedReader
import numpy as np

if __name__ == "__main__":
    rdr = WFSInterleavedReader(
        path=r"D:\Pipeline RUL Data\B.wfs",
        dtype=np.dtype("<i2"),   # little-endian int16
        n_channels=8,
        header_bytes=2,          # <- use probe result
        fs=1_000_000
    )
    print(rdr.info())
    total_cells = 0
    for start_idx, block in rdr.iter_cell_power(cell_size=500, seconds_per_chunk=10.0):
        # block shape: [8, n_cells]
        total_cells += block.shape[1]
        if start_idx == 0:
            # print a tiny sanity row
            print("first block shape:", block.shape, "min/max:", float(block.min()), float(block.max()))
            # stop early for demo
            break
    print("streaming OK; saw cells:", total_cells)
