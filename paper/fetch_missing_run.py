import wandb
import pandas as pd
import sys

api = wandb.Api()
print("Default entity:", api.default_entity)

try:
    projects = api.projects("swami2004")
    print("Projects for swami2004:", [p.name for p in projects])
except Exception as e:
    print("Error getting projects:", e)

# Try fetching runs
try:
    runs = api.runs("swami2004/REP")
    print(f"Found {len(runs)} runs in swami2004/REP")
    
    target_run = None
    # We look for "concave_truncated" and "123" in the name
    for r in runs:
        if "concave_truncated" in r.name and "__123__" in r.name:
            target_run = r
            break
            
    if target_run:
        print("Found target run:", target_run.name)
        history = target_run.history(samples=5000)
        metrics_to_fetch = [
            "eval/success_once",
            "eval/return",
            "train/return",
            "train/success_once"
        ]
        
        data = {"Step": history.get("_step", history.get("step", history.index))}
        for metric in metrics_to_fetch:
            if metric in history:
                col_name = f"{target_run.name} - {metric}"
                data[col_name] = history[metric]
        
        df = pd.DataFrame(data)
        df.to_csv("wandb_export_missing_run_123.csv", index=False)
        print("Saved missing run data.")
    else:
        print("Target run not found in the project's runs.")
except Exception as e:
    print("Error fetching runs:", e)
