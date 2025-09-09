# tune_similarity.py
import os, glob, yaml
import numpy as np
import pandas as pd
from similarity import predict_rul_for_test

# ---------- config ----------
FAST = True  # True: use only [30,50,70,90]; False: use all windows from config
GRID_saw   = [21, 31, 41]
GRID_step  = [1, 2]
GRID_topk  = [25, 50, 100]
GRID_pow   = [0.0, 1.0, 1.5, 2.0]
SCOPES     = ["same_pipe"]       # or ["same_pipe","all_pipes"] to compare
FUSION     = "median"            # "median" or "mean"
# --------------------------------

def load_cfg():
    with open("config.yml","r") as f:
        return yaml.safe_load(f)

def load_his(hi_dir):
    files = sorted(glob.glob(os.path.join(hi_dir, "*_HI.csv")))
    data = {}
    for p in files:
        df = pd.read_csv(p)
        pipe = os.path.basename(p).replace("_HI.csv","")
        data[pipe] = df
    return data

def build_refs(data, ref_sensors):
    refs_all = []
    for pipe, df in data.items():
        for s in ref_sensors:
            sdf = df[df.sensor==s].sort_values("t_sec")
            if len(sdf)==0: continue
            refs_all.append({"pipe": pipe, "sensor": int(s),
                             "t": sdf.t_sec.to_numpy(),
                             "hi": sdf.cum_hits.to_numpy()})
    return refs_all

def choose_refs(refs_all, pipe, scope):
    if scope == "same_pipe":
        return [r for r in refs_all if r["pipe"] == pipe]
    elif scope == "all_pipes":
        return refs_all
    return refs_all

def eval_combo(data, refs_all, test_sensors, pct_list, saw, step, topk, pw, scope, fusion):
    rows = []
    for pipe, df in data.items():
        refs = choose_refs(refs_all, pipe, scope) or refs_all
        for s in test_sensors:
            sdf = df[df.sensor==s].sort_values("t_sec")
            if len(sdf) < 8: 
                continue
            T_end = float(sdf.t_sec.iloc[-1])
            for pct in pct_list:
                t_cut = (pct/100.0)*T_end
                cut = sdf[sdf.t_sec<=t_cut]
                if len(cut) < 8:
                    continue
                pred = predict_rul_for_test(
                    cut.t_sec.to_numpy(), cut.cum_hits.to_numpy(), refs,
                    saw_n_points=saw,
                    ref_step_points=step,
                    top_k_segments=topk,
                    slope_weight_power=pw
                )
                rows.append({
                    "pipe": pipe, "sensor": int(s), "pct": int(pct),
                    "true": float(T_end - t_cut),
                    "pred": float(pred)
                })
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df["abs_err"] = (df["pred"] - df["true"]).abs()

    # per-sensor summary (not returned)
    # pipe-level fusion
    fuse = []
    for (pipe, pct), g in df.groupby(["pipe","pct"]):
        pred = float(g.pred.mean()) if fusion=="mean" else float(g.pred.median())
        true = float(g.true.median())
        fuse.append({"pipe": pipe, "pct": int(pct), "pred": pred, "true": true})
    fdf = pd.DataFrame(fuse)
    fdf["abs_err"] = (fdf["pred"] - fdf["true"]).abs()
    mae = float(fdf["abs_err"].mean())
    rmse = float(np.sqrt((fdf["abs_err"]**2).mean()))
    return df, {"MAE": mae, "RMSE": rmse}

def main():
    cfg = load_cfg()
    pct_list = [30,50,70,90] if FAST else [int(x) for x in cfg["similarity"]["test_percent_windows"]]
    test_sensors = set(cfg["sensor_map"]["test"])
    ref_sensors  = set(cfg["sensor_map"]["reference"])
    data = load_his(cfg["hi_output_dir"])
    refs_all = build_refs(data, ref_sensors)

    results = []
    best = None
    os.makedirs(cfg["models_output_dir"], exist_ok=True)

    for scope in SCOPES:
        for saw in GRID_saw:
            for step in GRID_step:
                for topk in GRID_topk:
                    for pw in GRID_pow:
                        _, score = eval_combo(
                            data, refs_all, test_sensors, pct_list,
                            saw, step, topk, pw, scope, FUSION
                        )
                        if score is None: 
                            continue
                        row = dict(scope=scope, saw=saw, step=step, topk=topk, pow=pw, fusion=FUSION, **score)
                        results.append(row)
                        if best is None or row["MAE"] < best["MAE"]:
                            best = row
                        print(f"scope={scope} saw={saw} step={step} topk={topk} pow={pw} -> MAE={score['MAE']:.2f} RMSE={score['RMSE']:.2f}")

    res = pd.DataFrame(results).sort_values("MAE")
    out_csv = os.path.join(cfg["models_output_dir"], "tuning_summary.csv")
    res.to_csv(out_csv, index=False)
    print("\nBest:", best)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
