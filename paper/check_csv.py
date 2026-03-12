import glob
import pandas as pd

dfs = []
for f in glob.glob("*.csv"):
    df = pd.read_csv(f)
    print(f"File {f}: columns {len(df.columns)}")
    for col in df.columns:
        if " - " in col and "MIN" not in col and "MAX" not in col:
            print("  ", col)
