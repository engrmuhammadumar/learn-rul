# tests_day10.py
import os, json, time, hashlib, platform, numpy as np
from cfar_core import ca_cfar_full, StreamingCFAR, alpha_from_f_quantile

SEED = 42
rng = np.random.default_rng(SEED)

def versions():
    import numpy, sys
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "numpy": numpy.__version__,
    }

def hash_config(cfg: dict) -> str:
    b = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]

def make_noise_cells(Ncells:int, L:int, sigma:float=0.2) -> np.ndarray:
    # simulate cell mean-power as average of squares over L samples
    x = sigma * rng.standard_normal((Ncells, L))
    return (x*x).mean(axis=1).astype(np.float32)

def inject_bursts(cp: np.ndarray, idxs, bumps):
    cp = cp.copy()
    for i,b in zip(idxs, bumps):
        if 0 <= i < len(cp): cp[i] += b
    return cp

def test_pfa_noise_only():
    print("\n[TEST] PFA on noise-only stream")
    L,T,G,pfa = 500, 20, 2, 1e-3
    Ncells = 2_000_000
    sigma = 0.2
    cp = make_noise_cells(Ncells, L, sigma)
    thr, det, lm, alpha, valid = ca_cfar_full(cp, L,T,G,pfa)
    emp_pfa = det.mean()
    print(f"alpha={alpha:.6f}, empirical PFA={emp_pfa:.4e}, target={pfa:g}")
    assert abs(emp_pfa - pfa) < 5e-4 or emp_pfa/pfa < 1.5, "PFA too far from target"

def test_streaming_equals_full_many_chunkings():
    print("\n[TEST] Streaming equals full-array for many random chunk sizes")
    L,T,G,pfa = 500, 20, 2, 1e-3
    Ncells = 1_000_000
    cp = make_noise_cells(Ncells, L)
    # add structured bumps to test detections around boundaries
    cp = inject_bursts(cp, [5_000, 123_456, 700_000, 999_500], [0.25, 0.30, 0.22, 0.35])

    thr_ref, det_ref, _, _, valid = ca_cfar_full(cp, L,T,G,pfa)

    for chunk in [7_777, 32_000, 64_000, 123_457, 250_001]:
        s = StreamingCFAR(L,T,G,pfa, chunk_size_cells=chunk)
        det = np.zeros_like(det_ref, bool)
        thr = np.zeros_like(thr_ref, float)
        for start, d, th in s.process_array(cp):
            det[start:start+len(d)] = d
            thr[start:start+len(th)] = th
        # compare only where both have valid training
        diff = np.count_nonzero((det != det_ref) & valid)
        print(f"chunk={chunk:6d}  diffs(valid)={diff}")
        assert diff == 0, f"streaming mismatch for chunk {chunk}"

def test_checkpoint_resume_matches_uninterrupted():
    print("\n[TEST] Resume checkpoint equals uninterrupted run")
    L,T,G,pfa = 500, 20, 2, 1e-3
    Ncells = 600_000
    cp = make_noise_cells(Ncells, L)
    cp = inject_bursts(cp, [55_000, 120_000, 250_123, 470_000], [0.3, 0.28, 0.35, 0.26])

    # uninterrupted
    sA = StreamingCFAR(L,T,G,pfa, chunk_size_cells=80_000)
    detA = np.zeros(Ncells, bool)
    for start, d, _ in sA.process_array(cp):
        detA[start:start+len(d)] = d

    # simulated interruption with checkpoint file
    ckpt = "cfar_ckpt.json"
    if os.path.exists(ckpt): os.remove(ckpt)
    sB = StreamingCFAR(L,T,G,pfa, chunk_size_cells=80_000, checkpoint_path=ckpt)
    detB = np.zeros(Ncells, bool)

    # process first three chunks then "crash"
    it = sB.process_array(cp)
    for _ in range(3):
        start, d, _ = next(it)
        detB[start:start+len(d)] = d
    # simulate process exit (sB saves automatically every 1M cells; we force-save here)
    sB._save()

    # "restart"
    sC = StreamingCFAR(L,T,G,pfa, chunk_size_cells=80_000, checkpoint_path=ckpt)
    for start, d, _ in sC.process_array(cp):
        detB[start:start+len(d)] = d
    if os.path.exists(ckpt): os.remove(ckpt)

    diff = np.count_nonzero(detA != detB)
    print(f"resume diffs={diff}")
    assert diff == 0, "resume path differs from uninterrupted"

def write_run_metadata(run_dir: str, cfg: dict):
    os.makedirs(run_dir, exist_ok=True)
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "versions": versions(),
        "seed": SEED,
        "config": cfg,
        "config_hash": hash_config(cfg),
    }
    with open(os.path.join(run_dir,"run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[meta] wrote", os.path.join(run_dir,"run_meta.json"))
    return meta

def demo_reproducible_run():
    print("\n[DEMO] Reproducible run record")
    L,T,G,pfa = 500, 20, 2, 1e-3
    cfg = {"L":L, "T":T, "G":G, "pfa":pfa, "chunk_cells": 123_457}
    run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    meta = write_run_metadata(run_dir, cfg)

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    test_pfa_noise_only()
    test_streaming_equals_full_many_chunkings()
    test_checkpoint_resume_matches_uninterrupted()
    demo_reproducible_run()

    print("\nALL TESTS PASSED ✅")
