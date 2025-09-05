# day9_chunk_invariance_test.py
import numpy as np
from day9_stream_cfar import StreamingCFAR, ca_cfar_same

def make_signal(Ncells=2_000_00, L=500, seed=1):
    rng = np.random.default_rng(seed)
    # base noise-only cell power (simulate as average of squares)
    x = (0.2 * rng.standard_normal(Ncells*L).reshape(Ncells, L)**2).mean(axis=1)
    # inject some bursts in cell power (raise a few cells)
    for idx in [5000, 123456, 700000, 1500000]:
        if idx < Ncells:
            x[idx] += 0.3
            x[idx+1:idx+4] += np.linspace(0.2, 0.05, 3)
    return x

def run_stream(cp, L=500, T=20, G=2, pfa=1e-3, chunk=123_457):
    s = StreamingCFAR(L, T, G, pfa, chunk_size_cells=chunk)
    det = np.zeros_like(cp, dtype=bool)
    thr = np.zeros_like(cp, dtype=float)
    for start, d, th in s.process_array(cp):
        det[start:start+len(d)] = d
        thr[start:start+len(th)] = th
    return det, thr

if __name__ == "__main__":
    L, T, G, pfa = 500, 20, 2, 1e-3
    Ncells = 2_000_000
    cp = make_signal(Ncells=Ncells, L=L)

    # Full-array reference
    thr_ref, det_ref, _, _ = ca_cfar_same(cp, L=L, T=T, G=G, pfa=pfa)

    # Two different chunkings
    det_A, thr_A = run_stream(cp, L,T,G,pfa, chunk=64_000)
    det_B, thr_B = run_stream(cp, L,T,G,pfa, chunk=137_777)

    # Compare
    same_A = np.array_equal(det_A, det_ref)
    same_B = np.array_equal(det_B, det_ref)
    print("Chunk=64k equals reference:", same_A)
    print("Chunk=137777 equals reference:", same_B)

    # Optional: Hamming distance
    diffA = np.count_nonzero(det_A != det_ref)
    diffB = np.count_nonzero(det_B != det_ref)
    print("Differences A:", diffA, "B:", diffB)
