"""
Launch reward variant sweep for the RewardGeom study.

Runs all combinations of reach × gate × terminal variants across tasks and seeds,
sequentially (one at a time) to fit within 8GB VRAM budget.

Usage:
    # Dry run — print all commands without executing
    python examples/reward_sweep.py --env_id PickCube-v1 --dry_run

    # Full Phase 1 sweep (PickCube only, all variants, 3 seeds)
    python examples/reward_sweep.py --env_id PickCube-v1

    # Baseline only (verify infrastructure)
    python examples/reward_sweep.py --env_id PickCube-v1 \\
        --reach_variants tanh --gate_variants hard --terminal_variants hard_jump

    # Single dimension: reach shape study
    python examples/reward_sweep.py --env_id PickCube-v1 \\
        --gate_variants hard --terminal_variants hard_jump

    # Multi-task sweep
    python examples/reward_sweep.py --env_ids PickCube-v1 PushCube-v1
"""

import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime

# Full variant lists
ALL_REACH_VARIANTS = [
    "tanh",
    "linear",
    "exponential",
    "quadratic",
    "piecewise_cc",
    "concave_truncated",
    "oscillatory",
    "sparse_threshold",
]
ALL_GATE_VARIANTS = ["hard", "soft", "additive", "curriculum"]
ALL_TERMINAL_VARIANTS = ["hard_jump", "smooth", "none"]
ALL_TASKS = ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PlaceSphere-v1"]
DEFAULT_SEEDS = [1, 2, 3]

# Per-task max steps (from @register_env decorator)
TASK_MAX_STEPS = {
    "PickCube-v1": 50,
    "PushCube-v1": 50,
    "StackCube-v1": 50,
    "PlaceSphere-v1": 50,
}


def build_command(
    env_id: str,
    reach_variant: str,
    gate_variant: str,
    terminal_variant: str,
    seed: int,
    total_timesteps: int,
    num_envs: int,
    track: bool,
    wandb_entity: str,
    wandb_project: str,
    wandb_group: str,
) -> list:
    exp_name = (
        f"{env_id}__reach-{reach_variant}__gate-{gate_variant}"
        f"__term-{terminal_variant}__s{seed}"
    )
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "baselines", "ppo", "ppo_fast.py"),
        "--env_id", env_id,
        "--reach_variant", reach_variant,
        "--gate_variant", gate_variant,
        "--terminal_variant", terminal_variant,
        "--seed", str(seed),
        "--total_timesteps", str(total_timesteps),
        "--num_envs", str(num_envs),
        "--exp_name", exp_name,
    ]
    if track:
        cmd += [
            "--track",
            "--wandb_entity", wandb_entity,
            "--wandb_project_name", wandb_project,
            "--wandb_group", wandb_group,
        ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="RewardGeom sweep launcher")
    parser.add_argument("--env_id", default="PickCube-v1",
                        help="Single task to sweep (overridden by --env_ids)")
    parser.add_argument("--env_ids", nargs="+", default=None,
                        help="Multiple tasks to sweep")
    parser.add_argument("--reach_variants", nargs="+", default=ALL_REACH_VARIANTS)
    parser.add_argument("--gate_variants", nargs="+", default=ALL_GATE_VARIANTS)
    parser.add_argument("--terminal_variants", nargs="+", default=ALL_TERMINAL_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--num_envs", type=int, default=256,
                        help="Parallel envs per run (256 fits 8GB VRAM)")
    parser.add_argument("--track", action="store_true", default=True,
                        help="Log to W&B (default: on)")
    parser.add_argument("--no_track", dest="track", action="store_false")
    parser.add_argument("--wandb_entity", default="swami2004")
    parser.add_argument("--wandb_project", default="REP")
    parser.add_argument("--wandb_group", default="REP")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip combos whose run directories already exist")
    args = parser.parse_args()

    tasks = args.env_ids if args.env_ids else [args.env_id]

    combos = list(itertools.product(
        tasks,
        args.reach_variants,
        args.gate_variants,
        args.terminal_variants,
        args.seeds,
    ))

    print(f"[RewardGeom Sweep] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tasks:      {tasks}")
    print(f"Reach:      {args.reach_variants}")
    print(f"Gate:       {args.gate_variants}")
    print(f"Terminal:   {args.terminal_variants}")
    print(f"Seeds:      {args.seeds}")
    print(f"Total runs: {len(combos)}")
    print(f"Steps/run:  {args.total_timesteps:,}")
    print(f"W&B track:  {args.track}")
    print()

    failed = []
    for i, (env_id, reach, gate, term, seed) in enumerate(combos):
        cmd = build_command(
            env_id=env_id,
            reach_variant=reach,
            gate_variant=gate,
            terminal_variant=term,
            seed=seed,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            track=args.track,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
        )

        run_label = (
            f"[{i+1}/{len(combos)}] {env_id} | "
            f"reach={reach} gate={gate} term={term} seed={seed}"
        )

        if args.dry_run:
            print(run_label)
            print("  " + " ".join(cmd))
            print()
            continue

        print(f"\n{'='*70}")
        print(run_label)
        print(f"{'='*70}")
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[WARN] Run exited with code {ret}: {run_label}")
            failed.append((env_id, reach, gate, term, seed))

    if not args.dry_run:
        print(f"\n[Done] {len(combos) - len(failed)}/{len(combos)} runs succeeded.")
        if failed:
            print("[Failed runs]:")
            for f in failed:
                print(f"  {f}")


if __name__ == "__main__":
    main()
