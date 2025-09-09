# plot_rul_curves.py
import os, yaml, re
import pandas as pd
import matplotlib.pyplot as plt

def read_cfg(path="config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pick_col(cols, *patterns, required=False, friendly="column"):
    """
    Pick the first column whose name matches any of the regex patterns (case-insensitive).
    """
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(c):
                return c
    if required:
        raise KeyError(
            f"Could not find {friendly}. Tried patterns: {patterns}. "
            f"Available columns: {list(cols)}"
        )
    return None

def coerce_numeric(s, name):
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        raise ValueError(f"Column '{name}' could not be converted to numeric.")
    return s

def prepare_sensor_df(df: pd.DataFrame):
    df = normalize_cols(df)

    # Likely names
    pipe_col   = pick_col(df.columns, r"^pipe$", r"^unit$", r"^id$", friendly="pipe id", required=True)
    sensor_col = pick_col(df.columns, r"^sensor$", r"sensor[_-]?id", r"^sid$", friendly="sensor id", required=True)
    pct_col    = pick_col(df.columns, r"^pct$", r"percent", r"%", r"life[_-]?used", friendly="percent life used", required=True)

    # True RUL column candidates
    true_col = pick_col(
        df.columns,
        r"^true$", r"^true[_-]?rul(_sec)?$", r"^rul[_-]?true$",
        r"^target[_-]?rul", r"^label[_-]?rul", r"^gt[_-]?rul",
        r"^rul$",  # often ground-truth RUL is just 'rul'
        friendly="ground-truth RUL column",
        required=True
    )

    # Predicted RUL candidates
    pred_col = pick_col(
        df.columns,
        r"^predicted[_-]?rul(_sec)?$", r"^rul[_-]?pred(icted)?(_sec)?$",
        r"^yhat[_-]?rul", r"^pred[_-]?rul", r"^rul[_-]?est",
        friendly="predicted RUL column",
        required=True
    )

    # Coerce numeric
    df[pct_col]  = coerce_numeric(df[pct_col], pct_col)
    df[true_col] = coerce_numeric(df[true_col], true_col)
    df[pred_col] = coerce_numeric(df[pred_col], pred_col)

    mapping = {
        "pipe": pipe_col, "sensor": sensor_col, "pct": pct_col,
        "true": true_col, "pred": pred_col
    }
    return df, mapping

def prepare_fused_df(df: pd.DataFrame):
    df = normalize_cols(df)

    pipe_col = pick_col(df.columns, r"^pipe$", r"^unit$", r"^id$", friendly="pipe id", required=True)
    pct_col  = pick_col(df.columns, r"^pct$", r"percent", r"%", r"life[_-]?used", friendly="percent life used", required=True)

    true_col = pick_col(
        df.columns,
        r"^true$", r"^true[_-]?rul(_sec)?$", r"^rul[_-]?true$",
        r"^target[_-]?rul", r"^label[_-]?rul", r"^gt[_-]?rul",
        r"^rul$",
        friendly="ground-truth RUL column",
        required=True
    )

    pred_col = pick_col(
        df.columns,
        r"^predicted[_-]?rul(_sec)?$", r"^rul[_-]?pred(icted)?(_sec)?$",
        r"^yhat[_-]?rul", r"^pred[_-]?rul", r"^rul[_-]?est",
        friendly="predicted RUL column",
        required=True
    )

    df[pct_col]  = coerce_numeric(df[pct_col], pct_col)
    df[true_col] = coerce_numeric(df[true_col], true_col)
    df[pred_col] = coerce_numeric(df[pred_col], pred_col)

    mapping = {"pipe": pipe_col, "pct": pct_col, "true": true_col, "pred": pred_col}
    return df, mapping

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    cfg = read_cfg("config.yml")
    out_dir = cfg["models_output_dir"]
    ensure_dir(out_dir)

    sens_csv  = os.path.join(out_dir, "rul_predictions.csv")
    fused_csv = os.path.join(out_dir, "rul_predictions_fused.csv")

    sens = pd.read_csv(sens_csv)
    sens_df, sm = prepare_sensor_df(sens)

    # Log the column mapping so you can see what was used
    print("[per-sensor column mapping]", sm)

    # per-sensor plots
    for (pipe, sensor), g in sens_df.groupby([sm["pipe"], sm["sensor"]]):
        g = g.sort_values(sm["pct"])
        plt.figure()
        plt.plot(g[sm["pct"]], g[sm["true"]], marker="o", label="True RUL")
        plt.plot(g[sm["pct"]], g[sm["pred"]], marker="s", label="Pred RUL")
        plt.title(f"{pipe} — S{sensor}")
        plt.xlabel("% life used")
        plt.ylabel("RUL (s)")
        plt.grid(True); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{pipe}_S{sensor}_RUL_curve.png"))
        plt.close()

    # fused (pipe level)
    if os.path.exists(fused_csv):
        fuse = pd.read_csv(fused_csv)
        fuse_df, fm = prepare_fused_df(fuse)
        print("[fused column mapping]", fm)

        for pipe, g in fuse_df.groupby(fm["pipe"]):
            g = g.sort_values(fm["pct"])
            plt.figure()
            plt.plot(g[fm["pct"]], g[fm["true"]], marker="o", label="True RUL")
            plt.plot(g[fm["pct"]], g[fm["pred"]], marker="s", label="Pred RUL")
            plt.title(f"{pipe} — fused")
            plt.xlabel("% life used")
            plt.ylabel("RUL (s)")
            plt.grid(True); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{pipe}_fused_RUL_curve.png"))
            plt.close()
    else:
        print(f"[info] No fused CSV at {fused_csv}; skipping fused plots.")

    print(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    main()
