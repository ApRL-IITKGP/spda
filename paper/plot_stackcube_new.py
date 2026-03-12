"""
Plot StackCube-v1 results from the two latest W&B CSV exports.
Produces:
  fig_stackcube_main.pdf       — mean ± 95% CI, all variants
  fig_stackcube_per_seed.pdf   — 3×1 grid by seed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 2.2,
    "axes.grid": True,
    "grid.alpha": 0.4,
})

CSV_FILES = [
    "/home/chomy/Downloads/wandb_export_2026-03-12T23_39_13.843+05_30.csv",
    "/home/chomy/Downloads/wandb_export_2026-03-12T23_38_33.834+05_30.csv",
]

VARIANTS_INCLUDE = {
    "tanh",
    "concave_truncated",
    "adaptive_tanh_episode",
    "adaptive_concave_distance",
}

DISPLAY_NAMES = {
    "tanh":                      r"$\mathtt{tanh}$",
    "concave_truncated":         r"$\mathtt{concave\_truncated}$",
    "adaptive_tanh_episode":     r"$\mathtt{adaptive\_tanh\_episode}$",
    "adaptive_concave_distance": r"$\mathtt{adaptive\_concave\_distance}$ (ours)",
}

# Color + linestyle: make "ours" pop
palette = sns.color_palette("husl", 8)
COLORS = {
    "tanh":                      palette[0],
    "concave_truncated":         palette[2],
    "adaptive_tanh_episode":     sns.color_palette("deep")[2],
    "adaptive_concave_distance": "#e63946",   # bold red for ours
}
LWIDTHS = {k: 2.0 for k in COLORS}
LWIDTHS["adaptive_concave_distance"] = 2.8

SEEDS = ["1", "42", "123"]


# ── Load ─────────────────────────────────────────────────────────────────────
def load_stackcube(metric="success_once"):
    rows = []
    for fpath in CSV_FILES:
        df = pd.read_csv(fpath)
        metric_cols = [
            c for c in df.columns
            if "StackCube" in c
            and f"eval/{metric}" in c
            and not c.endswith("__MIN")
            and not c.endswith("__MAX")
        ]
        for col in metric_cols:
            m = re.search(r"StackCube-v1__reach-(.+?)__gate.*?__ppo_fast__(\d+)__", col)
            if not m:
                continue
            variant = m.group(1)
            seed = m.group(2)
            if variant not in VARIANTS_INCLUDE:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            steps = pd.to_numeric(df["Step"], errors="coerce")
            mask = vals.notna() & steps.notna()
            for s, v in zip(steps[mask].values, vals[mask].values):
                rows.append({"step_M": s / 1e6, "value": v,
                             "variant": variant, "seed": seed})
    data = pd.DataFrame(rows)
    # EMA smooth per (variant, seed)
    data = data.sort_values(["variant", "seed", "step_M"])
    data["smooth"] = data.groupby(["variant", "seed"])["value"].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    return data


# ── Figure 1: mean ± 95% CI ───────────────────────────────────────────────
def plot_main(data, out="fig_stackcube_main.pdf"):
    fig, ax = plt.subplots(figsize=(8, 5))

    order = ["tanh", "concave_truncated", "adaptive_tanh_episode", "adaptive_concave_distance"]
    for variant in order:
        sub = data[data["variant"] == variant]
        if sub.empty:
            continue
        label = DISPLAY_NAMES[variant]
        color = COLORS[variant]
        lw = LWIDTHS[variant]
        sns.lineplot(
            data=sub, x="step_M", y="smooth",
            color=color, linewidth=lw, label=label,
            errorbar=("ci", 95), err_kws={"alpha": 0.15},
            ax=ax, legend=False,
        )

    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("eval/success\_once")
    ax.set_title("StackCube-v1: Eval Success Rate\n"
                 r"(EMA $\alpha{=}0.3$, shaded 95\% CI over 3 seeds)")
    ax.set_ylim(-0.02, 1.08)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Legend with bold entry for ours
    handles, labels = [], []
    for v in order:
        line = plt.Line2D([0], [0], color=COLORS[v], linewidth=LWIDTHS[v],
                          label=DISPLAY_NAMES[v])
        handles.append(line)
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: per-seed 3×1 ───────────────────────────────────────────────
def plot_per_seed(data, out="fig_stackcube_per_seed.pdf"):
    order = ["tanh", "concave_truncated", "adaptive_tanh_episode", "adaptive_concave_distance"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ix, seed in enumerate(SEEDS):
        ax = axes[ix]
        sub = data[data["seed"] == seed]
        for variant in order:
            vsub = sub[sub["variant"] == variant]
            if vsub.empty:
                continue
            ax.plot(
                vsub["step_M"], vsub["smooth"],
                color=COLORS[variant],
                linewidth=LWIDTHS[variant],
                label=DISPLAY_NAMES[variant],
            )
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Steps (M)")
        ax.set_ylim(-0.02, 1.08)
        if ix == 0:
            ax.set_ylabel("eval/success\_once")

    # Single shared legend on the right
    handles = [
        plt.Line2D([0], [0], color=COLORS[v], linewidth=LWIDTHS[v], label=DISPLAY_NAMES[v])
        for v in order
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(1.18, 0.95), framealpha=0.9)
    fig.suptitle("StackCube-v1: Per-Seed Eval Success Rate"
                 r" (EMA $\alpha{=}0.3$)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.chdir("/home/chomy/git/ManiSkill3/paper")

    data = load_stackcube("success_once")
    print("Loaded rows:", len(data))
    print(data.groupby(["variant", "seed"])["value"].agg(["max", "last"]).round(3))
    print()

    plot_main(data)
    plot_per_seed(data)
