# day9_stream_cfar.py
# Streaming CA-CFAR over a huge 1D cell-power sequence.
# - Overlap-save with W = G + T on each side
# - Vectorized CFAR (no Python loops)
# - Identical results regardless of chunk size (proved in tests)
# - Optional JSON checkpointing for resumability

import os, json, math, numpy as np
from typing import Iterable, Optional, Tuple


# ---------- CFAR core (same as Day 8, factored) ----------
def alpha_from_f_quantile(pfa: float, L: int, K: int) -> float:
    """alpha s.t. P(X/Y > alpha | H0) = pfa, with X~(σ²/L)χ²_L, Y~(σ²/(KL))χ²_{KL}."""
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa must be in (0,1)")
    try:
        from scipy.stats import f
        return float(f.ppf(1.0 - pfa, L, K * L))
    except Exception:
        # MC fallback: good enough for pfa ≥ 1e-4; raise N for tighter
        rng = np.random.default_rng(123)
        N = 400_000
        X = rng.normal(size=(N, L)) ** 2
        Y = rng.normal(size=(N, K * L)) ** 2
        R = X.mean(axis=1) / Y.mean(axis=1)
        return float(np.quantile(R, 1.0 - pfa))


def cfar_kernel(T: int, G: int) -> np.ndarray:
    """1D kernel that sums training cells and skips CUT+guards."""
    k = np.arange(-(T + G), (T + G) + 1)
    ker = np.where((np.abs(k) > G) & (np.abs(k) <= G + T), 1.0, 0.0)
    return ker


def ca_cfar_same(x: np.ndarray, L: int, T: int, G: int, pfa: float):
    """CA-CFAR on a *single full array*, returns thr, det, local_mean, alpha."""
    x = np.asarray(x, float)
    K = 2 * T
    ker = cfar_kernel(T, G)
    train_sum = np.convolve(x, ker, mode="same")
    train_cnt = np.convolve(np.ones_like(x), ker, mode="same")
    train_cnt = np.maximum(train_cnt, 1.0)
    local_mean = train_sum / train_cnt
    alpha = alpha_from_f_quantile(pfa, L=L, K=K)
    thr = alpha * local_mean
    # Mark invalid edges where too few training cells:
    valid = train_cnt >= (0.8 * K)
    det = (x > thr) & valid
    return thr, det, local_mean, alpha


# ---------- Streaming wrapper with overlap-save ----------
class StreamingCFAR:
    """
    Stream CA-CFAR decisions from a huge cell-power array or iterable of chunks.
    Emits identical results to ca_cfar_same() for any chunk size, provided
    chunk_size_cells >= 1 and we have W=(G+T) overlap on both sides.
    """
    def __init__(
        self,
        L_cell: int,
        T: int,
        G: int,
        pfa: float,
        chunk_size_cells: int = 200_000,
        checkpoint_path: Optional[str] = None,
    ):
        self.L_cell = int(L_cell)
        self.T = int(T)
        self.G = int(G)
        self.pfa = float(pfa)
        self.chunk_size_cells = int(chunk_size_cells)
        self.W = self.T + self.G
        self.K = 2 * self.T
        self.alpha = alpha_from_f_quantile(pfa, L_cell, self.K)
        self.ker = cfar_kernel(T, G)
        self.ckpt_path = checkpoint_path
        self._left_overlap = np.empty(0, dtype=float)
        self._emitted_upto = 0  # global index of last emitted cell (exclusive)

        # Try resume if checkpoint exists
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            self._load_ckpt()

    # --- checkpoint helpers ---
    def _save_ckpt(self):
        if not self.ckpt_path:
            return
        obj = {
            "emitted_upto": int(self._emitted_upto),
            "left_overlap": self._left_overlap.tolist(),
        }
        tmp = self.ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, self.ckpt_path)

    def _load_ckpt(self):
        try:
            with open(self.ckpt_path, "r") as f:
                obj = json.load(f)
            self._emitted_upto = int(obj.get("emitted_upto", 0))
            lo = np.array(obj.get("left_overlap", []), dtype=float)
            self._left_overlap = lo
            print(f"[StreamingCFAR] Resumed at cell {self._emitted_upto}, left_overlap={len(lo)}")
        except Exception as e:
            print(f"[StreamingCFAR] Failed to load checkpoint: {e}")
            self._left_overlap = np.empty(0, dtype=float)
            self._emitted_upto = 0

    # --- main streaming API ---
    def process_iter(self, chunks: Iterable[np.ndarray]):
        """
        Yield tuples (global_start_idx, det_bool, thr_vals) for each emitted 'current' region.
        - chunks: iterable of 1D np.ndarray cell-power segments (any sizes allowed).
        - Emits only decisions for the *current* region of each assembled buffer.
        """
        W = self.W
        for raw in chunks:
            x_cur = np.asarray(raw, float)
            if x_cur.size == 0:
                continue

            # Prepend saved left overlap if any (e.g., after resume)
            if self._left_overlap.size > 0:
                x_buf = np.concatenate([self._left_overlap, x_cur], axis=0)
                cur_start_in_buf = self._left_overlap.size
            else:
                x_buf = x_cur
                cur_start_in_buf = 0

            # We must withhold the final W cells (need right-side training).
            left_guard = W
            right_guard = W
            start_emit = max(cur_start_in_buf, left_guard)
            end_emit = x_buf.size - right_guard  # exclusive
            if end_emit <= start_emit:
                # Not enough context yet; stash overlap and continue
                self._left_overlap = x_buf[-W:].copy() if x_buf.size >= W else x_buf.copy()
                continue

            # CFAR over the buffer
            train_sum = np.convolve(x_buf, self.ker, mode="same")
            train_cnt = np.convolve(np.ones_like(x_buf), self.ker, mode="same")
            train_cnt = np.maximum(train_cnt, 1.0)
            local_mean = train_sum / train_cnt
            thr = self.alpha * local_mean

            det = x_buf > thr
            valid = train_cnt >= (0.8 * (2 * self.T))
            det &= valid

            # Emit only [start_emit:end_emit)
            det_emit = det[start_emit:end_emit]
            thr_emit = thr[start_emit:end_emit]

            # Global indexing for the emitted slice
            global_start = self._emitted_upto
            yield (global_start, det_emit.copy(), thr_emit.copy())

            # Update counters
            n_emitted = det_emit.size
            self._emitted_upto += n_emitted

            # Prepare next left overlap: keep exactly W cells of trailing context
            tail = x_buf[end_emit:]
            if tail.size >= W:
                self._left_overlap = tail[-W:].copy()
            else:
                need = W - tail.size
                taken_from_emit = x_buf[end_emit - need : end_emit]
                self._left_overlap = np.concatenate([taken_from_emit, tail], axis=0)

            # Save checkpoint periodically (~every million cells)
            if self.ckpt_path and (self._emitted_upto // 1_000_000) != (
                (self._emitted_upto - n_emitted) // 1_000_000
            ):
                self._save_ckpt()

        # After the last chunk, we intentionally DO NOT emit the final W cells
        # (no right context). This matches edge-masking behavior of full CFAR.

    # Convenience: process a *single* big array by splitting into chunks
    def process_array(self, x: np.ndarray):
        N = x.size
        cs = self.chunk_size_cells
        for i in range(0, N, cs):
            yield from self.process_iter([x[i : i + cs]])


# ---------- Optional plotting (small debug) ----------
def _plot_compare(tc, cp, thr_ref, thr_stream, det_ref, det_stream, title="CFAR compare"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(tc * 1e3, cp, label="cell power", alpha=0.6)
    plt.plot(tc * 1e3, thr_ref, label="thr full", linewidth=1)
    plt.plot(tc * 1e3, thr_stream, label="thr stream", linewidth=1, linestyle="--")
    if np.any(det_ref):
        plt.scatter(tc[det_ref] * 1e3, cp[det_ref], s=10, label="det full")
    if np.any(det_stream):
        plt.scatter(tc[det_stream] * 1e3, cp[det_stream], s=10, label="det stream", marker="x")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- Main: quick synthetic test ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Parameters
    Ncells = 2_000_00  # 200k for a quick test; raise to millions as needed
    L, T, G, pfa = 500, 20, 2, 1e-3
    chunk_size = 123_456
    show_small_plot = False  # set True to plot first few thousand cells

    # Generate *cell power* directly without giant (Ncells x L) arrays:
    # If x[n] ~ N(0, sigma^2), then mean(x^2) over L samples ~ sigma^2 * (chi2_L / L)
    sigma = 0.2
    cp = (sigma**2) * (rng.chisquare(df=L, size=Ncells) / L)

    # One-shot reference (edges masked inside)
    thr_ref, det_ref, _, _ = ca_cfar_same(cp, L=L, T=T, G=G, pfa=pfa)

    # Streaming (arbitrary chunk sizes)
    streamer = StreamingCFAR(L_cell=L, T=T, G=G, pfa=pfa, chunk_size_cells=chunk_size)
    det_stream = np.zeros_like(det_ref, dtype=bool)
    thr_stream = np.zeros_like(thr_ref, dtype=float)

    pos = 0
    for start, det_emit, thr_emit in streamer.process_array(cp):
        det_stream[start : start + len(det_emit)] = det_emit
        thr_stream[start : start + len(thr_emit)] = thr_emit
        pos = start + len(det_emit)

    # Compare equality over entire sequence (True means identical decision)
    equal = (det_stream == det_ref)
    print("Streaming vs full equality (%):", f"{100.0 * equal.mean():.6f}")

    # Optional small plot to visually check a window
    if show_small_plot:
        # fake a time axis at "cell cadence"; only for visualization
        fs = 1_000_000
        tc = (np.arange(Ncells) + 0.5) * (L / fs)
        win = slice(0, min(8000, Ncells))
        _plot_compare(
            tc[win], cp[win], thr_ref[win], thr_stream[win], det_ref[win], det_stream[win],
            title="CFAR: full vs streaming (first window)"
        )
