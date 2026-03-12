import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 300,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.5,
})

def plot_metric(metric_name, out_name, title, ylabel):
    data = []
    
    # Load data from all CSV files matching wandb_export*.csv
    for f in glob.glob("wandb_export_*.csv"):
        df = pd.read_csv(f)
        step_cols = [c for c in df.columns if c.lower() == 'step']
        if not step_cols:
            continue
        step_col = step_cols[0]
    
        for col in df.columns:
            if metric_name not in col:
                continue
            
            # Parse col name for PushCube only
            if "PushCube" not in col:
                continue
                
            match = re.search(r'reach-(.*?)__gate-.*?__ppo_fast__(\d+)__', col)
            if not match:
                continue
            variant = match.group(1)
            seed = match.group(2)
            
            # Map variants
            if variant == "concave_truncated":
                v_name = "concave_truncated (R=0.2)"
            elif variant == "concave_truncated_10":
                v_name = "concave_truncated (R=0.1)"
            else:
                v_name = variant
                
            s_col = df[step_col].values
            val_col = df[col].values
            
            # Drop NaNs
            mask = ~np.isnan(val_col)
            
            for s, v in zip(s_col[mask], val_col[mask]):
                data.append({
                    "Step": s,
                    "Value": v,
                    "Variant": v_name,
                    "Seed": seed
                })
    
    if not data:
        print(f"No data found for metric {metric_name}")
        return

    plot_df = pd.DataFrame(data)
    plot_df["Step"] = plot_df["Step"] / 1e6  # convert to Millions
    
    smooth_df = plot_df.sort_values(by=["Variant", "Seed", "Step"])
    # Exponential moving average smoothing for better visuals
    smooth_df["Value"] = smooth_df.groupby(["Variant", "Seed"])["Value"].transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    
    # Palette definition
    palette = sns.color_palette("husl", 8)
    colors = {
        "tanh": palette[0],
        "tanh_sq": palette[1],
        "concave_truncated (R=0.2)": palette[2],
        "concave_truncated (R=0.1)": palette[3],
        "adaptive_tanh_episode": sns.color_palette("deep")[2], # distinct green
        "adaptive_concave_linear": palette[5],
        "adaptive_concave_threshold": palette[6],
    }
    
    # --- FIG MAIN ---
    variants_main = [
        "tanh", 
        "adaptive_tanh_episode"
    ]
    df_main = smooth_df[smooth_df["Variant"].isin(variants_main)]
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_main, x="Step", y="Value", hue="Variant", style="Variant",
                 errorbar=('ci', 95), dashes=False, palette=colors)
    plt.title(f"{title} - Key Reach Variants\n(EMA Smoothed $\\alpha=0.3$, Shaded 95% CI over 3 seeds)")
    plt.xlabel("Environment Steps (Millions)")
    plt.ylabel(ylabel)
    plt.legend(title="Reach Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_name}_main.pdf")
    plt.close()
    
    # --- FIG PER SEED ---
    variants_per_seed = [
        "tanh",
        "adaptive_tanh_episode"
    ]
    df_seed = smooth_df[smooth_df["Variant"].isin(variants_per_seed)]
    seeds = ["1", "42", "123"] # Specific order
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    
    # Store global legend info
    all_handles, all_labels = [], []
    
    for ix, seed in enumerate(seeds):
        ax = axes[ix]
        df_s = df_seed[df_seed["Seed"] == seed]
        if len(df_s) > 0:
            sns.lineplot(data=df_s, x="Step", y="Value", hue="Variant", style="Variant", 
                         ax=ax, legend=True, palette=colors, dashes=False)
            
            # Extract handles to build a unified legend
            h, l = ax.get_legend_handles_labels()
            for hnd, lbl in zip(h, l):
                if lbl not in all_labels and lbl != "Variant":
                    all_labels.append(lbl)
                    all_handles.append(hnd)
            
            # Remove the sub-axis legend
            if ax.get_legend() is not None:
                ax.get_legend().remove()
                
        ax.set_title(f"Seed {seed} {'(Hard)' if seed=='1' else ''}")
        ax.set_xlabel("Steps (M)")
        if ix == 0:
            ax.set_ylabel(ylabel)
        
    fig.legend(all_handles, all_labels, title="Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_name}_per_seed.pdf")
    plt.close()

# Evaluate standard success (at end for PushCube)
plot_metric(" - eval/success_at_end", "fig_pushcube", "Eval Success at End", "Eval Success Rate")

# The user asked for eval returns, train returns, train success rate mapping. 
# We'll plot them with their specific file modifiers.
plot_metric(" - eval/return", "fig_pushcube_eval_return", "Eval Return", "Evaluation Return")
plot_metric(" - train/return", "fig_pushcube_train_return", "Train Return", "Training Return")
plot_metric(" - train/success_once", "fig_pushcube_train_success", "Train Success Once", "Training Success Rate")

print("Plots generated successfully using all wandb_export*.csv files for all target metrics!")
