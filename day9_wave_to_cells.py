# day9_wave_to_cells.py  (skeleton)
import numpy as np
from typing import Iterator, Tuple

def iter_cells_from_interleaved_wfs(path: str, n_channels=8, dtype=np.int16,
                                    fs=1_000_000, cell_size=500,
                                    chunk_seconds=10.0, header_bytes=16, little_endian=True
                                   ) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yields (global_cell_start_idx, cell_power_block) where:
      - cell_power_block has shape [n_channels, n_cells_in_block]
      - global_cell_start_idx is the starting cell index for this block
    This handles:
      - arbitrary file sizes
      - header bytes to skip
      - leftover partial cells carried to next chunk
    """
    dt = np.dtype(dtype).newbyteorder('<' if little_endian else '>')
    bytes_per_sample = dt.itemsize
    # number of interleaved samples to read per chunk
    samples_per_chunk = int(chunk_seconds * fs)
    # ensure chunk is a multiple of n_channels
    samples_per_chunk = (samples_per_chunk // n_channels) * n_channels

    with open(path, "rb") as f:
        f.seek(header_bytes)
        # carry samples per channel that didn't fit last cell
        carry = [np.empty((0,), dtype=dt) for _ in range(n_channels)]
        cell_index = 0

        while True:
            raw = np.fromfile(f, dtype=dt, count=samples_per_chunk)
            if raw.size == 0:
                break
            # reshape interleaved -> [samples, channels] then split per channel
            S = raw.size // n_channels
            raw = raw[:S*n_channels].reshape(S, n_channels)
            # de-interleave
            chans = [raw[:, ch].astype(np.float32) for ch in range(n_channels)]
            # prepend carry and compute whole cells
            cell_blocks = []
            n_cells_block = None
            for ch in range(n_channels):
                x = np.concatenate([carry[ch], chans[ch]], axis=0)
                # compute how many complete cells we can form
                n_cells = x.size // cell_size
                used = n_cells * cell_size
                if n_cells_block is None:
                    n_cells_block = n_cells
                else:
                    n_cells_block = min(n_cells_block, n_cells)  # equalize across channels
                # stash truncated channel back (we'll re-truncate to min across channels)
                carry[ch] = x  # temporarily

            # now build consistent block with same number of cells for all channels
            if n_cells_block and n_cells_block > 0:
                block = np.empty((n_channels, n_cells_block), dtype=np.float32)
                for ch in range(n_channels):
                    x = carry[ch]
                    used = n_cells_block * cell_size
                    seg = x[:used].reshape(n_cells_block, cell_size)
                    block[ch] = (seg.astype(np.float32)**2).mean(axis=1)
                    carry[ch] = x[used:]  # keep leftover for next chunk
                yield (cell_index, block)
                cell_index += n_cells_block

        # done; leftover partial cells in 'carry' are ignored (edges)
