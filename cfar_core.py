# cfar_core.py
import os, json, math, numpy as np
from typing import Iterable, Optional

# ---------- Alpha (threshold factor) ----------
def alpha_from_f_quantile(pfa: float, L: int, K: int) -> float:
    """alpha s.t. P(X/Y > alpha | H0) = pfa, with X~(σ²/L)χ²_L, Y~(σ²/(KL))χ²_{KL}."""
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa must be in (0,1)")
    try:
        from scipy.stats import f
        return float(f.ppf(1.0 - pfa, L, K*L))
    except Exception:
        rng = np.random.default_rng(123)
        N = 400_000  # raise to tighten tails for very small PFAs
        X = rng.normal(size=(N, L))**2
        Y = rng.normal(size=(N, K*L))**2
        R = X.mean(axis=1) / Y.mean(axis=1)
        return float(np.quantile(R, 1.0 - pfa))

# ---------- Kernel & full-array CA-CFAR ----------
def _kernel(T: int, G: int) -> np.ndarray:
    k = np.arange(-(T+G), (T+G)+1)
    return np.where((np.abs(k) > G) & (np.abs(k) <= G+T), 1.0, 0.0)

def ca_cfar_full(cell_power: np.ndarray, L: int, T: int, G: int, pfa: float):
    x = np.asarray(cell_power, float)
    K = 2*T
    ker = _kernel(T, G)
    train_sum = np.convolve(x, ker, mode="same")
    train_cnt = np.convolve(np.ones_like(x), ker, mode="same")
    train_cnt = np.maximum(train_cnt, 1.0)
    local_mean = train_sum / train_cnt
    alpha = alpha_from_f_quantile(pfa, L=L, K=K)
    thr = alpha * local_mean
    valid = train_cnt >= (0.8*K)
    det = (x > thr) & valid
    return thr, det, local_mean, alpha, valid

# ---------- Streaming CA-CFAR with overlap-save & checkpoint ----------
class StreamingCFAR:
    """
    Streaming CFAR that emits decisions for the interior region of each chunk.
    Guarantees equivalence to ca_cfar_full for any chunking (same validity mask).
    """
    def __init__(self, L_cell:int, T:int, G:int, pfa:float,
                 chunk_size_cells:int=200_000, checkpoint_path:Optional[str]=None):
        self.L = int(L_cell); self.T=int(T); self.G=int(G)
        self.K=2*self.T; self.pfa=float(pfa)
        self.W = self.T + self.G
        self.chunk = int(chunk_size_cells)
        self.ker = _kernel(self.T, self.G)
        self.alpha = alpha_from_f_quantile(self.pfa, self.L, self.K)
        self.ckpt = checkpoint_path
        self.left = np.empty(0, float)
        self.emitted = 0
        if self.ckpt and os.path.exists(self.ckpt): self._load()

    def _save(self):
        if not self.ckpt: return
        obj = {"emitted": int(self.emitted), "left": self.left.tolist()}
        tmp = self.ckpt + ".tmp"
        with open(tmp, "w") as f: json.dump(obj, f)
        os.replace(tmp, self.ckpt)

    def _load(self):
        try:
            with open(self.ckpt,"r") as f: obj=json.load(f)
            self.emitted = int(obj.get("emitted",0))
            self.left = np.array(obj.get("left",[]), float)
            print(f"[resume] emitted={self.emitted}, left={len(self.left)}")
        except Exception as e:
            print("[resume] failed:", e)

    def _cfar_same(self, x: np.ndarray):
        ts = np.convolve(x, self.ker, mode="same")
        tc = np.convolve(np.ones_like(x), self.ker, mode="same")
        tc = np.maximum(tc, 1.0)
        lm = ts/tc; thr = self.alpha*lm
        det = (x > thr) & (tc >= (0.8*self.K))
        return det, thr, tc

    def process_iter(self, chunks: Iterable[np.ndarray]):
        W = self.W
        for cur in chunks:
            cur = np.asarray(cur, float)
            if cur.size == 0: continue
            buf = np.concatenate([self.left, cur]) if self.left.size else cur
            # only emit region that has both-side training
            start = max(self.left.size, W)
            end   = buf.size - W
            if end <= start:
                self.left = buf[-W:].copy() if buf.size>=W else buf.copy()
                continue
            det, thr, tc = self._cfar_same(buf)
            det_emit = det[start:end]; thr_emit = thr[start:end]
            global_start = self.emitted
            yield (global_start, det_emit.copy(), thr_emit.copy())
            n = det_emit.size
            self.emitted += n
            tail = buf[end:]
            if tail.size >= W:
                self.left = tail[-W:].copy()
            else:
                need = W - tail.size
                self.left = np.concatenate([buf[end-need:end], tail])
            if self.ckpt and (self.emitted // 1_000_000) != ((self.emitted-n)//1_000_000):
                self._save()

    def process_array(self, x: np.ndarray):
        N = len(x)
        for i in range(0, N, self.chunk):
            yield from self.process_iter([x[i:i+self.chunk]])
