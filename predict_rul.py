# predict_rul.py
import os, glob, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from similarity import predict_rul_for_test, derivative_convolution

def load_hi_csv(path):
    return pd.read_csv(path)

def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean())/s if s>0 else np.zeros_like(x)

def pick_best_refs_in_pipe(pipe_df, test_sensor, exclude, saw_n_points, K=2):
    """
    Choose K reference sensors inside the same pipe that best match the test sensor's
    full-lifetime HI shape (in derivative-convolution, z-normalized). Excludes given sensors.
    """
    candidates = sorted(set(pipe_df["sensor"].unique()) - set([test_sensor]) - set(exclude))
    sdf_t = pipe_df[pipe_df["sensor"]==test_sensor].sort_values("t_sec")
    if len(sdf_t) < 8 or not candidates:
        return []
    t_t = sdf_t["t_sec"].to_numpy()
    hi_t = sdf_t["cum_hits"].to_numpy()
    dc_t = _z(derivative_convolution(hi_t, saw_n_points))
    L = len(dc_t)

    dists = []
    for s in candidates:
        sdf = pipe_df[pipe_df["sensor"]==s].sort_values("t_sec")
        if len(sdf) < L:
            continue
        # align by truncation (same length)
        hi = sdf["cum_hits"].to_numpy()[:L]
        dc = _z(derivative_convolution(hi, saw_n_points))
        d = np.linalg.norm(dc_t - dc)
        dists.append((d, int(s)))

    if not dists:
        return []
    dists.sort(key=lambda x: x[0])
    return [s for _, s in dists[:max(1, int(K))]]

def main():
    with open("config.yml","r") as f:
        cfg = yaml.safe_load(f)

    hi_dir   = cfg["hi_output_dir"]
    out_dir  = cfg["models_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    ref_sensors_cfg  = set(cfg["sensor_map"]["reference"])   # still used for other scopes
    test_sensors     = set(cfg["sensor_map"]["test"])

    sim_cfg = cfg.get("similarity", {})
    saw_n       = int(sim_cfg.get("saw_kernel_n_points", 41))
    steppts     = int(sim_cfg.get("ref_segment_step_points", 1))
    topk        = int(sim_cfg.get("top_k_segments", 25))
    slope_pow   = float(sim_cfg.get("slope_weight_power", 0.0))
    scope       = sim_cfg.get("reference_scope", "same_pipe").lower()
    auto_k      = int(sim_cfg.get("auto_ref_k", 2))
    auto_excl   = [int(x) for x in sim_cfg.get("auto_ref_exclude", [4,5])]
    pct_list    = [int(x) for x in sim_cfg.get(
        "test_percent_windows",
        [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
    )]
    make_plots  = bool(cfg.get("plots", True))
    fusion_method = cfg.get("fusion", {}).get("method", "median").lower()

    # load HIs
    hi_files = sorted(glob.glob(os.path.join(hi_dir, "*_HI.csv")))
    if not hi_files:
        raise RuntimeError(f"No HI CSVs in {hi_dir}. Run prepare_data.py first.")
    pipe_data = {}
    for p in hi_files:
        df = load_hi_csv(p)
        pipe = os.path.basename(p).replace("_HI.csv","")
        pipe_data[pipe] = df

    # prebuild references for non-auto scopes
    all_references = []
    for pipe, df in pipe_data.items():
        for s in ref_sensors_cfg:
            sdf = df[df["sensor"] == s].sort_values("t_sec")
            if len(sdf) == 0:
                continue
            all_references.append({
                "pipe": pipe, "sensor": int(s),
                "t": sdf["t_sec"].to_numpy(),
                "hi": sdf["cum_hits"].to_numpy()
            })

    sensor_preds = []
    for pipe, df in pipe_data.items():
        # refs selection per pipe
        if scope == "same_pipe":
            # use configured ref sensors from the same pipe
            base_refs = []
            for s in ref_sensors_cfg:
                sdf = df[df["sensor"]==s].sort_values("t_sec")
                if len(sdf)==0: continue
                base_refs.append({"pipe": pipe, "sensor": int(s),
                                  "t": sdf["t_sec"].to_numpy(),
                                  "hi": sdf["cum_hits"].to_numpy()})
        elif scope == "all_pipes":
            base_refs = all_references
        elif scope == "auto_in_pipe":
            base_refs = None   # will pick per test sensor
        else:
            base_refs = all_references

        for s in test_sensors:
            sdf = df[df["sensor"] == s].sort_values("t_sec")
            if len(sdf) < 8:
                continue
            T_end = float(sdf["t_sec"].iloc[-1])

            # pick refs for this test sensor if auto scope
            if scope == "auto_in_pipe":
                best_ref_sensors = pick_best_refs_in_pipe(df, s, exclude=auto_excl, saw_n_points=saw_n, K=auto_k)
                refs = []
                for rs in best_ref_sensors:
                    rsdf = df[df["sensor"]==rs].sort_values("t_sec")
                    if len(rsdf)==0: continue
                    refs.append({"pipe": pipe, "sensor": int(rs),
                                 "t": rsdf["t_sec"].to_numpy(),
                                 "hi": rsdf["cum_hits"].to_numpy()})
                # fallback: if none found, use all configured refs in same pipe or all pipes
                if not refs:
                    refs = [r for r in all_references if r["pipe"] == pipe] or all_references
            else:
                refs = [r for r in base_refs if (scope!="same_pipe" or r["pipe"]==pipe)]

            for pct in pct_list:
                t_cut = (pct/100.0) * T_end
                cut_df = sdf[sdf["t_sec"] <= t_cut]
                if len(cut_df) < 8:
                    continue

                t  = cut_df["t_sec"].to_numpy()
                hi = cut_df["cum_hits"].to_numpy()

                pred_sec = predict_rul_for_test(
                    t, hi, refs,
                    saw_n_points=saw_n,
                    ref_step_points=steppts,
                    top_k_segments=topk,
                    slope_weight_power=slope_pow
                )

                row = {
                    "pipe": pipe, "sensor": int(s), "pct": int(pct),
                    "t_cut_sec": float(t_cut),
                    "predicted_rul_sec": float(pred_sec),
                    "true_rul_sec": float(T_end - t_cut)
                }
                sensor_preds.append(row)

                if make_plots and pct in (30,50,70,90):
                    plt.figure()
                    plt.plot(sdf["t_sec"], sdf["cum_hits"], label="Full HI")
                    plt.plot(cut_df["t_sec"], cut_df["cum_hits"], label="In-service HI")
                    plt.axvline(t_cut, linestyle="--")
                    plt.title(f"{pipe} S{s} @ {pct}% | Pred RUL={pred_sec:.1f}s  True={T_end-t_cut:.1f}s")
                    plt.xlabel("Time (s)"); plt.ylabel("Cumulative AE hits")
                    plt.legend(); plt.tight_layout()
                    os.makedirs(out_dir, exist_ok=True)
                    plt.savefig(os.path.join(out_dir, f"{pipe}_S{s}_pct{pct}_traj.png"))
                    plt.close()

    sens_df = pd.DataFrame(sensor_preds)
    sens_csv = os.path.join(out_dir, "rul_predictions.csv")
    sens_df.to_csv(sens_csv, index=False)
    print(f"Wrote predictions: {sens_csv}")

    # pipe-level fusion
    fuse_rows = []
    for (pipe, pct), g in sens_df.groupby(["pipe","pct"]):
        pred = float(g["predicted_rul_sec"].median()) if fusion_method!="mean" else float(g["predicted_rul_sec"].mean())
        true = float(g["true_rul_sec"].median())
        fuse_rows.append({"pipe": pipe, "pct": int(pct), "predicted_rul_sec": pred, "true_rul_sec": true})
    fusion_df = pd.DataFrame(fuse_rows).sort_values(["pipe","pct"])
    fusion_csv = os.path.join(out_dir, "rul_predictions_fused.csv")
    fusion_df.to_csv(fusion_csv, index=False)
    print(f"Wrote pipe-level fused predictions: {fusion_csv}")

if __name__ == "__main__":
    main()
