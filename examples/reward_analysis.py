"""
Aggregate and visualize RewardGeom sweep results from W&B.

Usage:
    python examples/reward_analysis.py --env_id PickCube-v1
    python examples/reward_analysis.py --env_id PickCube-v1 --metric eval/success_once
    python examples/reward_analysis.py --env_id PickCube-v1 --output_dir figs/
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENTITY = "swami2004"
PROJECT = "REP"
GROUP = "REP"

REACH_VARIANTS = [
    "tanh", "linear", "exponential", "quadratic",
    "piecewise_cc", "concave_truncated", "oscillatory", "sparse_threshold",
]
GATE_VARIANTS = ["hard", "soft", "additive", "curriculum"]
TERMINAL_VARIANTS = ["hard_jump", "smooth", "none"]

PRIMARY_METRIC = "eval/success_once"
SECONDARY_METRICS = [
    "eval/return",
    "charts/explained_variance",
    "losses/grad_norm",
    "losses/value_loss",
]


def fetch_runs(env_id: str, api=None):
    """Fetch all RewardGeom runs for a given task from W&B."""
    import wandb
    if api is None:
        api = wandb.Api()

    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"group": GROUP, "config.env_id": env_id},
    )

    records = []
    for run in runs:
        cfg = run.config
        history = run.history(
            keys=[PRIMARY_METRIC] + SECONDARY_METRICS,
            pandas=True,
            samples=200,
        )
        records.append({
            "run_id": run.id,
            "name": run.name,
            "reach_variant": cfg.get("reach_variant", "tanh"),
            "gate_variant": cfg.get("gate_variant", "hard"),
            "terminal_variant": cfg.get("terminal_variant", "hard_jump"),
            "seed": cfg.get("seed", 1),
            "history": history,
            "summary": run.summary._json_dict,
        })
    return records


def convergence_step(history: pd.DataFrame, metric: str, threshold: float = 0.8) -> float:
    """Return the global_step at which metric first exceeds threshold (or NaN)."""
    col = metric.replace("/", ".")
    if col not in history.columns:
        col = metric
    if col not in history.columns:
        return float("nan")
    above = history[history[col] >= threshold]
    if above.empty:
        return float("nan")
    step_col = "_step" if "_step" in history.columns else history.columns[0]
    return above[step_col].iloc[0]


def aggregate_by_dimension(records: list, dimension: str, metric: str) -> dict:
    """Group runs by a single variant dimension, aggregate mean ± std over seeds."""
    groups = defaultdict(list)
    for r in records:
        key = r[dimension]
        val = r["summary"].get(metric, float("nan"))
        groups[key].append(val)

    agg = {}
    for k, vals in groups.items():
        vals = [v for v in vals if not np.isnan(v)]
        agg[k] = {
            "mean": np.mean(vals) if vals else float("nan"),
            "std": np.std(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return agg


def plot_dimension_comparison(
    records: list,
    dimension: str,
    metric: str = PRIMARY_METRIC,
    title: str = "",
    output_path: str = None,
):
    """Bar chart comparing variants along one dimension."""
    agg = aggregate_by_dimension(records, dimension, metric)
    keys = list(agg.keys())
    means = [agg[k]["mean"] for k in keys]
    stds = [agg[k]["std"] for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.2), 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(keys)))
    bars = ax.bar(keys, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xlabel(dimension.replace("_", " ").title())
    ax.set_ylabel(metric.split("/")[-1].replace("_", " ").title())
    ax.set_title(title or f"{metric} by {dimension}")
    ax.set_ylim(0, min(1.05, max(means) * 1.3 + 0.05) if means else 1.0)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_learning_curves(
    records: list,
    group_by: str = "reach_variant",
    metric: str = PRIMARY_METRIC,
    title: str = "",
    output_path: str = None,
):
    """Plot mean learning curves grouped by a dimension, with std band."""
    groups = defaultdict(list)
    for r in records:
        key = r[group_by]
        h = r["history"]
        step_col = "_step" if "_step" in h.columns else h.columns[0]
        metric_col = metric.replace("/", ".") if metric.replace("/", ".") in h.columns else metric
        if metric_col not in h.columns:
            continue
        groups[key].append((h[step_col].values, h[metric_col].values))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))

    for (key, curves), color in zip(sorted(groups.items()), colors):
        # Interpolate all curves to a common step grid
        all_steps = np.concatenate([c[0] for c in curves])
        step_grid = np.linspace(all_steps.min(), all_steps.max(), 200)
        interp_curves = []
        for steps, vals in curves:
            interp = np.interp(step_grid, steps, vals)
            interp_curves.append(interp)
        arr = np.array(interp_curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(step_grid, mean, label=key, color=color)
        ax.fill_between(step_grid, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Global Step")
    ax.set_ylabel(metric.split("/")[-1].replace("_", " ").title())
    ax.set_title(title or f"{metric} — grouped by {group_by}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def print_summary_table(records: list, metric: str = PRIMARY_METRIC):
    """Print a ranked table of all (reach × gate × terminal) combinations."""
    rows = []
    for r in records:
        rows.append({
            "reach": r["reach_variant"],
            "gate": r["gate_variant"],
            "terminal": r["terminal_variant"],
            "seed": r["seed"],
            "success_once": r["summary"].get(metric, float("nan")),
            "convergence_step": convergence_step(r["history"], PRIMARY_METRIC),
        })
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["reach", "gate", "terminal"])
        .agg(
            mean_success=("success_once", "mean"),
            std_success=("success_once", "std"),
            mean_conv_step=("convergence_step", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
        .sort_values("mean_success", ascending=False)
    )
    print("\n=== RewardGeom Summary Table ===")
    print(grouped.to_string(index=False, float_format="{:.3f}".format))
    return grouped


def main():
    parser = argparse.ArgumentParser(description="RewardGeom results analysis")
    parser.add_argument("--env_id", default="PickCube-v1")
    parser.add_argument("--metric", default=PRIMARY_METRIC)
    parser.add_argument("--output_dir", default="figs/reward_geom")
    parser.add_argument("--no_fetch", action="store_true",
                        help="Skip W&B fetch (use cached CSV if available)")
    parser.add_argument("--cache_csv", default=None,
                        help="Path to cache/load run data as CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch from W&B
    if not args.no_fetch:
        import wandb
        api = wandb.Api()
        print(f"Fetching runs for {args.env_id} from W&B ({ENTITY}/{PROJECT})...")
        records = fetch_runs(args.env_id, api=api)
        print(f"Found {len(records)} runs.")
    else:
        print("--no_fetch: skipping W&B fetch. Provide --cache_csv to load data.")
        return

    if not records:
        print("No runs found. Run the sweep first.")
        return

    # Summary table
    summary_df = print_summary_table(records, metric=args.metric)
    summary_df.to_csv(
        os.path.join(args.output_dir, f"{args.env_id}_summary.csv"), index=False
    )

    # Bar charts per dimension
    for dim in ["reach_variant", "gate_variant", "terminal_variant"]:
        plot_dimension_comparison(
            records,
            dimension=dim,
            metric=args.metric,
            title=f"{args.env_id} — {args.metric} by {dim}",
            output_path=os.path.join(args.output_dir, f"{args.env_id}_{dim}_bar.png"),
        )

    # Learning curves per dimension
    for dim in ["reach_variant", "gate_variant", "terminal_variant"]:
        plot_learning_curves(
            records,
            group_by=dim,
            metric=args.metric,
            title=f"{args.env_id} — learning curves by {dim}",
            output_path=os.path.join(args.output_dir, f"{args.env_id}_{dim}_curves.png"),
        )

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
