# env_check.py
import sys, platform
import numpy as np, pandas as pd, matplotlib
print("Python:", sys.version.split()[0])
print("Platform:", platform.system(), platform.release())
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("OK: imports worked.")
