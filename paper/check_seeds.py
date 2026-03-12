import pandas as pd
import glob
import re
import numpy as np

data = []
for f in glob.glob("wandb_export_*.csv"):
    df = pd.read_csv(f)
    step_cols = [c for c in df.columns if c.lower() == 'step']
    if not step_cols: continue
    step_col = step_cols[0]
    
    for col in df.columns:
        if " - eval/success_once" not in col: continue
        if "PickCube" not in col: continue
            
        match = re.search(r'reach-(.*?)__gate-.*?__ppo_fast__(\d+)__', col)
        if not match: continue
        variant = match.group(1)
        seed = match.group(2)
        
        if variant == "concave_truncated": v_name = "concave_truncated (R=0.2)"
        elif variant == "concave_truncated_10": v_name = "concave_truncated (R=0.1)"
        else: v_name = variant
        
        s_col = df[step_col].values
        val_col = df[col].values
        mask = ~np.isnan(val_col)
        
        if np.sum(mask) > 0:
            data.append({"Variant": v_name, "Seed": seed, "Count": np.sum(mask)})

df_counts = pd.DataFrame(data)
print(df_counts.groupby(["Variant"])["Seed"].unique().reset_index())
