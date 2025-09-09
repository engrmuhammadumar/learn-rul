# quick_check_hi.py
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./hi_outputs/B_HI.csv")
for s in [4,5,6,7]:
    sdf = df[df.sensor==s]
    plt.plot(sdf.t_sec, sdf.cum_hits, label=f"S{s}")
plt.legend(); plt.xlabel("Time (s)"); plt.ylabel("Cumulative AE hits"); plt.title("B: HI")
plt.show()
