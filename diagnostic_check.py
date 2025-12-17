"""
Diagnostic script to check actual file naming pattern
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path("F:/phm_rul_reserach/phm data")

print("Checking file naming patterns...")
print("="*80)

# Check if path exists
if not DATA_PATH.exists():
    print(f"❌ Path not found: {DATA_PATH}")
    exit()

# List first 10 CSV files
csv_files = sorted(list(DATA_PATH.glob("c_1_*.csv")))[:10]

print(f"\nFound {len(list(DATA_PATH.glob('c_1_*.csv')))} total c_1_*.csv files")
print(f"\nFirst 10 files:")
for f in csv_files:
    print(f"  • {f.name}")

# Try to load one file
if csv_files:
    print(f"\n\nTrying to load: {csv_files[0].name}")
    try:
        df = pd.read_csv(csv_files[0])
        print(f"✓ Successfully loaded!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Check wear file
wear_file = DATA_PATH / "c1_wear_processed.csv"
print(f"\n\nChecking wear file: {wear_file.name}")
if wear_file.exists():
    wear_df = pd.read_csv(wear_file)
    print(f"✓ Wear file exists")
    print(f"  Shape: {wear_df.shape}")
    print(f"  First few cuts: {wear_df['cut'].head(10).tolist()}")
else:
    print(f"❌ Wear file not found")

print("\n" + "="*80)