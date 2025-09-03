# analyze_cfar_csv.py
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "cfar_outputs"   # where day11_reader_cfar_run.py wrote files
PIPE = "B"                 # change if needed

def load_all(pipe=PIPE):
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"{pipe}_S*_cfar_*.csv")))
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        dfs.append(df)
    if not dfs:
        raise SystemExit(f"No CSVs found in {OUT_DIR}")
    return pd.concat(dfs, ignore_index=True), files

def main():
    df, files = load_all()
    print("Analyzed files:")
    for f in files:
        print("  -", f)

    # Basic stats
    total_cells = len(df)
    total_dets  = int(df["det"].sum())
    # print sensors as plain ints (not numpy.int64 objects)
    sensors = sorted(int(s) for s in df["sensor"].unique())
    duration_s = df["time_s"].max() - df["time_s"].min()
    if duration_s <= 0:
        duration_s = df["time_s"].max()  # if started near 0

    print(f"\nCells emitted (all sensors): {total_cells:,}")
    print(f"Detections (all sensors):    {total_dets:,}")
    print(f"Empirical PFA (noise-only baseline) ≈ {total_dets/total_cells:.3e}  "
          f"(target was whatever you set in the run)")
    print(f"Sensors present: {sensors}")
    print(f"Slice duration ≈ {duration_s:.3f} s")

    # Per-sensor breakdown
    g = df.groupby("sensor")["det"].agg(["count", "sum"])
    g["emp_PFA"] = g["sum"] / g["count"]
    print("\nPer-sensor:")
    print(g)

    # Timeline plot: detections per second (all sensors combined)
    bins = np.arange(np.floor(df["time_s"].min()), np.ceil(df["time_s"].max()) + 1.0, 1.0)
    per_sec = np.histogram(df.loc[df["det"] == 1, "time_s"], bins=bins)[0]
    t_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(8, 3))
    # Removed use_line_collection kwarg (not supported in your Matplotlib)
    markerline, stemlines, baseline = plt.stem(t_centers, per_sec)
    plt.xlabel("Time (s)")
    plt.ylabel("Detections per sec")
    plt.title("CFAR detections per second (all sensors)")
    plt.tight_layout()
    plt.show()

    # Optional: threshold sanity (pick one sensor)
    sid = sensors[0]
    df1 = df[df["sensor"] == sid]
    plt.figure(figsize=(9, 3))
    plt.plot(df1["time_s"], df1["thr"], lw=1)
    y = df1.loc[df1["det"] == 1, "thr"]
    x = df1.loc[df1["det"] == 1, "time_s"]
    if len(x):
        plt.scatter(x, y, s=10, label="detections")
    plt.xlabel("Time (s)")
    plt.ylabel("Threshold")
    plt.title(f"Sensor {sid}: threshold vs time (markers at detections)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
