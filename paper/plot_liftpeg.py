import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

palette = sns.color_palette("husl", 8)
colors = {
    "tanh": palette[0],
    "adaptive_tanh_episode": sns.color_palette("deep")[2],
}

df = pd.read_csv("wandb_liftpeg.csv")
df["seed"] = df["seed"].astype(str)
df["step_M"] = df["step"] / 1e6

def plot_metric(metric_name, out_name, title, ylabel):
    sub = df[df["metric"] == metric_name].copy()
    if sub.empty:
        print(f"No data for {metric_name}")
        return

    sub = sub.sort_values(["reach_variant", "seed", "step_M"])
    sub["value"] = sub.groupby(["reach_variant", "seed"])["value"].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )

    # --- FIG MAIN ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=sub, x="step_M", y="value", hue="reach_variant", style="reach_variant",
                 errorbar=('ci', 95), dashes=False, palette=colors)
    plt.title(f"LiftPegUpright-v1: {title}\n(EMA Smoothed $\\alpha=0.3$, Shaded 95% CI over 3 seeds)")
    plt.xlabel("Environment Steps (Millions)")
    plt.ylabel(ylabel)
    plt.legend(title="Reach Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_name}_main.pdf")
    plt.close()

    # --- FIG PER SEED ---
    seeds = ["1", "42", "123"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    all_handles, all_labels = [], []

    for ix, seed in enumerate(seeds):
        ax = axes[ix]
        df_s = sub[sub["seed"] == seed]
        if len(df_s) > 0:
            sns.lineplot(data=df_s, x="step_M", y="value", hue="reach_variant", style="reach_variant",
                         ax=ax, legend=True, palette=colors, dashes=False)
            h, l = ax.get_legend_handles_labels()
            for hnd, lbl in zip(h, l):
                if lbl not in all_labels and lbl != "reach_variant":
                    all_labels.append(lbl)
                    all_handles.append(hnd)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        ax.set_title(f"Seed {seed} {'(Hard)' if seed == '1' else ''}")
        ax.set_xlabel("Steps (M)")
        if ix == 0:
            ax.set_ylabel(ylabel)

    fig.legend(all_handles, all_labels, title="Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_name}_per_seed.pdf")
    plt.close()
    print(f"Saved {out_name}_main.pdf and {out_name}_per_seed.pdf")

plot_metric("eval/success_once",  "fig_liftpeg",              "Eval Success Once",   "Eval Success Rate")
plot_metric("eval/success_at_end","fig_liftpeg_success_end",  "Eval Success at End", "Eval Success Rate")
plot_metric("eval/return",        "fig_liftpeg_eval_return",  "Eval Return",         "Evaluation Return")
plot_metric("train/return",       "fig_liftpeg_train_return", "Train Return",        "Training Return")
plot_metric("train/success_once", "fig_liftpeg_train_success","Train Success Once",  "Training Success Rate")
