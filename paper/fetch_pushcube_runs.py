import wandb
import pandas as pd
import sys

api = wandb.Api()
entity = "swami2004"
project = "REP"

try:
    # Fetch all runs from the project, sort by created_at descending
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    
    pushcube_runs = []
    # Find the latest 6 PushCube runs
    for r in runs:
        if "PushCube" in r.name:
            pushcube_runs.append(r)
        if len(pushcube_runs) == 6:
            break
            
    if len(pushcube_runs) < 6:
        print(f"Warning: Only found {len(pushcube_runs)} PushCube runs.")
        
    if not pushcube_runs:
        print("No PushCube runs found.")
        sys.exit(1)
        
    print(f"Found {len(pushcube_runs)} PushCube runs:")
    for r in pushcube_runs:
        print(" -", r.name)
        
    metrics_to_fetch = [
        "eval/success_once",
        "eval/success_at_end",
        "eval/return",
        "train/return",
        "train/success_once"
    ]
    
    # We will just export them into one large CSV for plotting later
    all_data = []
    for r in pushcube_runs:
        print(f"Fetching history for {r.name}...")
        history = r.history(samples=5000)
        
        step_col = history.get("_step", history.get("step", history.index))
        data = {"Step": step_col}
        
        for metric in metrics_to_fetch:
            if metric in history:
                col_name = f"{r.name} - {metric}"
                data[col_name] = history[metric]
        
        all_data.append(pd.DataFrame(data))
        
    # Merge all run dataframes on Step
    if all_data:
        from functools import reduce
        final_df = reduce(lambda left, right: pd.merge(left, right, on="Step", how="outer"), all_data)
        out_file = "wandb_export_pushcube.csv"
        final_df.to_csv(out_file, index=False)
        print(f"Saved PushCube run data to {out_file}")

except Exception as e:
    print("Error fetching runs:", e)
