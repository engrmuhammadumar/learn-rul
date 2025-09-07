# day11_memmap_reader.py
import os, numpy as np
from typing import Iterator, Tuple, Sequence, Optional

class WFSInterleavedReader:
    """
    Memory-mapped reader for headered, interleaved multi-channel waveform.
    - dtype: numpy dtype (e.g., np.dtype('<i2') for little-endian int16)
    - n_channels: number of sensors
    - header_bytes: skip at start
    - fs: sampling rate (Hz), used only for time calculations
    """
    def __init__(self, path:str, dtype:np.dtype, n_channels:int,
                 header_bytes:int=0, fs:int=1_000_000):
        self.path = path
        self.dtype = dtype
        self.nch = int(n_channels)
        self.header = int(header_bytes)
        self.fs = int(fs)

        self._size = os.path.getsize(path)
        if self.header >= self._size:
            raise ValueError("header_bytes >= file size")

        # total scalar samples (all channels interleaved)
        total_vals = (self._size - self.header) // self.dtype.itemsize
        if total_vals % self.nch != 0:
            raise ValueError(f"File size not divisible by channel count: "
                             f"total_vals={total_vals}, n_channels={self.nch}")
        self.samples_per_ch = total_vals // self.nch
        self.duration_s = self.samples_per_ch / self.fs

        # Build a memmap of the data region as a flat array
        self._mm = np.memmap(self.path, mode="r", dtype=self.dtype, offset=self.header, shape=(total_vals,))

    def info(self) -> str:
        return (f"WFSInterleavedReader(path={self.path}, dtype={self.dtype.str}, "
                f"nch={self.nch}, header={self.header} B, samples/ch={self.samples_per_ch:,}, "
                f"duration={self.duration_s:.2f} s)")

    def iter_cell_power(self, cell_size:int=500, seconds_per_chunk:float=10.0
                       ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yield (global_cell_start_idx, block) where block has shape [n_channels, n_cells].
        Reads ~seconds_per_chunk, carries partial cells across chunk boundaries.
        """
        if cell_size <= 0: raise ValueError("cell_size must be > 0")
        fs = self.fs
        nch = self.nch
        # how many *frames* (interleaved sets of nch samples) per chunk
        frames_per_chunk = int(seconds_per_chunk * fs)
        # make frames_per_chunk a multiple of cell_size for simple alignment
        frames_per_chunk = (frames_per_chunk // cell_size) * cell_size
        if frames_per_chunk <= 0:
            frames_per_chunk = cell_size * 2000  # default ~1s if tiny sec/chunk given

        total_frames = self.samples_per_ch
        # We will step across the flat memmap in strides of nch samples (frames)
        # Mapping: frame i occupies indices [i*nch, (i+1)*nch)
        # We'll read contiguous ranges of frames, then reshape to (frames, nch)

        carry = [np.empty((0,), dtype=np.float32) for _ in range(nch)]
        out_cell_idx = 0

        # use int64 to avoid overflow on huge files
        n_frames_done = np.int64(0)
        while n_frames_done < total_frames:
            need = int(min(frames_per_chunk, total_frames - n_frames_done))
            # slice flat memmap for this chunk (scalar samples)
            start_scalar = int(n_frames_done * nch)
            end_scalar   = int((n_frames_done + need) * nch)
            flat = self._mm[start_scalar:end_scalar]

            # de-interleave -> (frames, nch)
            frames = flat.reshape(need, nch).astype(np.float32)

            # For each channel: prepend carry, form whole cells
            # Equalize to the min available n_cells across channels
            nch_cells = []
            min_cells = None
            for ch in range(nch):
                x = np.concatenate([carry[ch], frames[:, ch]], axis=0)
                n_cells = x.size // cell_size
                if min_cells is None or n_cells < min_cells:
                    min_cells = n_cells
                nch_cells.append(x)

            if min_cells and min_cells > 0:
                block = np.empty((nch, min_cells), dtype=np.float32)
                used = min_cells * cell_size
                for ch in range(nch):
                    x = nch_cells[ch]
                    seg = x[:used].reshape(min_cells, cell_size)
                    # mean square (cell power)
                    block[ch] = (seg * seg).mean(axis=1)
                    carry[ch] = x[used:]
                yield (out_cell_idx, block)
                out_cell_idx += min_cells
            else:
                # accumulate more frames
                for ch in range(nch):
                    carry[ch] = nch_cells[ch]

            n_frames_done += need

        # Done. We intentionally drop leftover partial cells in 'carry' (edge).
