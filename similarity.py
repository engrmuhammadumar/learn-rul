# similarity.py  (HI-point domain with top-K and slope weighting)
import numpy as np

def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    if s == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / s

def saw_kernel(n:int):
    if n < 3:
        n = 3
    k = np.linspace(1.0, -1.0, n)
    k = k / (np.abs(k).sum() + 1e-12)
    return k

def derivative_convolution(x: np.ndarray, n:int):
    dx = np.gradient(np.asarray(x, dtype=float))
    k = saw_kernel(n)
    return np.convolve(dx, k, mode="same")

def segment_reference_points(traj: np.ndarray, step_points:int, test_len:int):
    segs = []
    N = len(traj)
    last_start = max(0, N - test_len)
    step = max(1, int(step_points))
    for start in range(0, last_start + 1, step):
        end = start + test_len
        segs.append((start, traj[start:end]))
    return segs

def _gaussian_weights(distances: np.ndarray):
    # robust scale from median
    if len(distances) == 0:
        return distances
    s = np.median(distances) + 1e-9
    w = np.exp(- (distances / s)**2)
    ssum = w.sum()
    return w / ssum if ssum > 0 else np.ones_like(w)/len(w)

def predict_rul_for_test(test_t: np.ndarray, test_hi: np.ndarray,
                         references: list,             # {'pipe','sensor','t','hi'}
                         saw_n_points:int,
                         ref_step_points:int,
                         top_k_segments:int = 50,
                         slope_weight_power: float = 1.0):
    """
    Fuzzy-weighted RUL using derivative-convolution on HI trajectories (HI-point domain).
    Uses top-K closest segments and multiplies distance-weights by late-life slope weight.
    """
    test_dc = _z(derivative_convolution(test_hi, saw_n_points))
    L = len(test_dc)
    if L < 5:
        return np.nan

    distances, ref_ruls, slope_mag = [], [], []

    for ref in references:
        ref_t, ref_hi = ref["t"], ref["hi"]
        ref_dc = _z(derivative_convolution(ref_hi, saw_n_points))

        # precompute slope d(hi)/dt for late-life emphasis
        dhi_dt = np.gradient(ref_hi, ref_t)

        for start_idx, seg in segment_reference_points(ref_dc, ref_step_points, L):
            # distance in DC domain
            d = np.linalg.norm(test_dc - seg)
            distances.append(d)

            # RUL from the end of the segment to the reference end (seconds)
            seg_end_idx = start_idx + L - 1
            seg_end_time = ref_t[seg_end_idx]
            ref_end_time = ref_t[-1]
            ref_ruls.append(ref_end_time - seg_end_time)

            # slope magnitude at the segment end index
            slope_mag.append(abs(dhi_dt[seg_end_idx]))

    if not distances:
        return np.nan

    distances = np.asarray(distances, float)
    ref_ruls  = np.asarray(ref_ruls,  float)
    slope_mag = np.asarray(slope_mag, float)

    # keep the K closest segments
    K = min(int(top_k_segments), len(distances))
    idx = np.argpartition(distances, K-1)[:K]
    dK, rK, sK = distances[idx], ref_ruls[idx], slope_mag[idx]

    # weights: distance * (normalized slope^power)
    w_dist = _gaussian_weights(dK)
    s_norm = sK / (sK.max() + 1e-12)
    w_slope = np.power(s_norm, max(0.0, float(slope_weight_power)))
    w = w_dist * (w_slope if slope_weight_power > 0 else 1.0)
    w_sum = w.sum()
    if w_sum == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w_sum

    return float((w * rK).sum())
