# os_cfar_streaming.py
import os, json, numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Iterable, Optional

def _train_mask(T: int, G: int) -> np.ndarray:
    W = T + G
    idx = np.arange(2*W + 1)
    # keep training cells; drop CUT (center) and guards
    return (np.abs(idx - W) > G)

def alpha_os_cfar(pfa: float, L: int, K: int, q: float,
                  N: int = 600_000, seed: int = 123) -> float:
    """
    Calibrate α so that P( X / Q_q(training) > α | H0 ) = pfa,
    with X and each training cell ~ (1/L)*χ²_L (Gaussian noise cell-power).
    Monte Carlo; increase N for tighter tails.
    """
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N) / L
    Y = rng.chisquare(df=L, size=(N, K)) / L
    qv = np.quantile(Y, q, axis=1)
    R = X / qv
    return float(np.quantile(R, 1.0 - pfa))

class StreamingOSCFAR:
    """
    Overlap-save streaming OS-CFAR over a 1D cell-power stream.
    Emits (global_start_idx, det_bool, thr_vals) for each emitted 'current' region.
    Chunk-invariant just like the CA-CFAR version.
    """
    def __init__(self, L_cell: int, T: int, G: int, pfa: float, q: float = 0.7,
                 chunk_size_cells: int = 200_000,
                 checkpoint_path: Optional[str] = None,
                 alpha_cache_path: Optional[str] = "os_cfar_alpha_cache.json"):
        self.L = int(L_cell); self.T = int(T); self.G = int(G)
        self.K = 2*self.T; self.pfa = float(pfa); self.q = float(q)
        self.W = self.T + self.G
        self.chunk = int(chunk_size_cells)
        self.mask = _train_mask(self.T, self.G)
        self.ckpt = checkpoint_path
        self.left = np.empty(0, float)
        self.emitted = 0
        # α cache (so we don't recalibrate every run)
        key = f"L={self.L}|K={self.K}|q={self.q}|pfa={self.pfa}"
        self.alpha = None
        if alpha_cache_path and os.path.exists(alpha_cache_path):
            try:
                with open(alpha_cache_path, "r") as f:
                    cache = json.load(f)
                if key in cache:
                    self.alpha = float(cache[key])
            except Exception:
                pass
        if self.alpha is None:
            self.alpha = alpha_os_cfar(self.pfa, self.L, self.K, self.q)
            if alpha_cache_path:
                try:
                    cache = {}
                    if os.path.exists(alpha_cache_path):
                        with open(alpha_cache_path, "r") as f:
                            cache = json.load(f)
                    cache[key] = float(self.alpha)
                    tmp = alpha_cache_path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(cache, f, indent=2)
                    os.replace(tmp, alpha_cache_path)
                except Exception:
                    pass

        if self.ckpt and os.path.exists(self.ckpt):
            self._load()

    # ----- checkpointing -----
    def _save(self):
        if not self.ckpt: return
        obj = {"emitted": int(self.emitted), "left": self.left.tolist()}
        tmp = self.ckpt + ".tmp"
        with open(tmp, "w") as f: json.dump(obj, f)
        os.replace(tmp, self.ckpt)

    def _load(self):
        try:
            with open(self.ckpt, "r") as f:
                obj = json.load(f)
            self.emitted = int(obj.get("emitted", 0))
            self.left = np.array(obj.get("left", []), float)
            print(f"[OS-CFAR resume] emitted={self.emitted}, left={len(self.left)}")
        except Exception as e:
            print("[OS-CFAR resume] failed:", e)

    # ----- main streaming API -----
    def process_iter(self, chunks: Iterable[np.ndarray]):
        W = self.W; K = self.K
        q = self.q; alpha = self.alpha; mask = self.mask

        for cur in chunks:
            x_cur = np.asarray(cur, float)
            if x_cur.size == 0: continue
            buf = np.concatenate([self.left, x_cur]) if self.left.size else x_cur

            # valid centers are [W, len(buf)-W)
            centers_start = W
            centers_end   = len(buf) - W
            if centers_end <= centers_start:
                self.left = buf[-W:].copy() if len(buf) >= W else buf.copy()
                continue

            # sliding windows and training-only view
            win = sliding_window_view(buf, window_shape=2*W+1)  # shape (N-2W, 2W+1)
            train = win[:, mask]                                 # shape (N-2W, K)

            # fast per-row quantile using partition (index, no interpolation)
            # idx for q-quantile in 0..K-1
            qi = int(np.floor(q * (K - 1)))
            part = np.partition(train, qi, axis=1)
            qv = part[:, qi]
            thr_core = alpha * qv

            # emit only for the current region that has both-side context:
            start_emit = max(self.left.size, W)
            end_emit   = len(buf) - W
            # index into thr_core (thr for center i maps to global cell i)
            i0 = start_emit - W
            i1 = end_emit   - W
            thr_emit = thr_core[i0:i1]
            det_emit = buf[start_emit:end_emit] > thr_emit

            # yield
            global_start = self.emitted
            yield (global_start, det_emit.copy(), thr_emit.copy())
            n = det_emit.size
            self.emitted += n

            # prepare next left overlap
            tail = buf[end_emit:]
            if tail.size >= W:
                self.left = tail[-W:].copy()
            else:
                need = W - tail.size
                self.left = np.concatenate([buf[end_emit-need:end_emit], tail])

            # periodic checkpoint
            if self.ckpt and (self.emitted // 1_000_000) != ((self.emitted - n) // 1_000_000):
                self._save()

    def process_array(self, x: np.ndarray):
        N = len(x)
        for i in range(0, N, self.chunk):
            yield from self.process_iter([x[i:i+self.chunk]])
